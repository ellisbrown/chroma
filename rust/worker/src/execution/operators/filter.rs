use std::{
    collections::{BTreeMap, HashMap},
    ops::{BitAnd, BitOr, Bound},
};

use chroma_error::{ChromaError, ErrorCodes};
use chroma_index::metadata::types::MetadataIndexError;
use chroma_types::{
    BooleanOperator, Chunk, DirectDocumentComparison, DirectWhereComparison, DocumentOperator,
    MaterializedLogOperation, MetadataSetValue, MetadataValue, PrimitiveOperator, SetOperator,
    SignedRoaringBitmap, Where, WhereChildren, WhereComparison,
};
use roaring::RoaringBitmap;
use thiserror::Error;
use tonic::async_trait;
use tracing::{trace, Instrument, Span};

use crate::{
    execution::operator::Operator,
    segment::{
        metadata_segment::MetadataSegmentReader, LogMaterializer, LogMaterializerError,
        MaterializedLogRecord,
    },
};

use super::{
    fetch_log::FetchLogOutput,
    fetch_segment::{FetchSegmentError, FetchSegmentOutput},
};

#[derive(Clone, Debug)]
pub struct FilterOperator {
    pub query_ids: Option<Vec<String>>,
    pub where_clause: Option<Where>,
}

#[derive(Debug)]
pub struct FilterInput {
    pub logs: FetchLogOutput,
    pub segments: FetchSegmentOutput,
}

#[derive(Debug)]
pub struct FilterOutput {
    pub log_offset_ids: SignedRoaringBitmap,
    pub compact_offset_ids: SignedRoaringBitmap,
}

#[derive(Error, Debug)]
pub enum FilterError {
    #[error("Error processing fetch segment output: {0}")]
    FetchSegment(#[from] FetchSegmentError),
    #[error("Error reading metadata index: {0}")]
    Index(#[from] MetadataIndexError),
    #[error("Error materializing log: {0}")]
    LogMaterializer(#[from] LogMaterializerError),
}

impl ChromaError for FilterError {
    fn code(&self) -> ErrorCodes {
        match self {
            FilterError::FetchSegment(e) => e.code(),
            FilterError::Index(e) => e.code(),
            FilterError::LogMaterializer(e) => e.code(),
        }
    }
}

/// This sturct provides an abstraction over the materialized logs that is similar to the metadata segment
pub(crate) struct MetadataLogReader<'me> {
    // This maps metadata keys to `BTreeMap`s, which further map values to offset ids
    // This mimics the layout in the metadata segment
    // //TODO: Maybe a sorted vector with binary search is more lightweight and performant?
    compact_metadata: HashMap<&'me str, BTreeMap<&'me MetadataValue, RoaringBitmap>>,
    // This maps offset ids to documents, excluding deleted ones
    document: HashMap<u32, &'me str>,
    // This contains all existing offset ids that are touched by the logs
    updated_offset_ids: RoaringBitmap,
    // This maps user ids to offset ids, excluding deleted ones
    user_id_to_offset_id: HashMap<&'me str, u32>,
}

impl<'me> MetadataLogReader<'me> {
    pub(crate) fn new(logs: &'me Chunk<MaterializedLogRecord<'me>>) -> Self {
        let mut compact_metadata: HashMap<_, BTreeMap<&MetadataValue, RoaringBitmap>> =
            HashMap::new();
        let mut document = HashMap::new();
        let mut updated_offset_ids = RoaringBitmap::new();
        let mut user_id_to_offset_id = HashMap::new();
        for (log, _) in logs.iter() {
            if !matches!(
                log.final_operation,
                MaterializedLogOperation::Initial | MaterializedLogOperation::AddNew
            ) {
                updated_offset_ids.insert(log.offset_id);
            }
            if !matches!(
                log.final_operation,
                MaterializedLogOperation::DeleteExisting
            ) {
                user_id_to_offset_id.insert(log.merged_user_id_ref(), log.offset_id);
                let log_metadata = log.merged_metadata_ref();
                for (key, val) in log_metadata.into_iter() {
                    compact_metadata
                        .entry(key)
                        .or_default()
                        .entry(val)
                        .or_default()
                        .insert(log.offset_id);
                }
                if let Some(doc) = log.merged_document_ref() {
                    document.insert(log.offset_id, doc);
                }
            }
        }
        Self {
            compact_metadata,
            document,
            updated_offset_ids,
            user_id_to_offset_id,
        }
    }
    pub(crate) fn get(
        &self,
        key: &str,
        val: &MetadataValue,
        op: &PrimitiveOperator,
    ) -> Result<RoaringBitmap, FilterError> {
        if let Some(metadata_value_to_offset_ids) = self.compact_metadata.get(key) {
            let bounds = match op {
                PrimitiveOperator::Equal => (Bound::Included(&val), Bound::Included(&val)),
                PrimitiveOperator::GreaterThan => (Bound::Excluded(&val), Bound::Unbounded),
                PrimitiveOperator::GreaterThanOrEqual => (Bound::Included(&val), Bound::Unbounded),
                PrimitiveOperator::LessThan => (Bound::Unbounded, Bound::Excluded(&val)),
                PrimitiveOperator::LessThanOrEqual => (Bound::Unbounded, Bound::Included(&val)),
                PrimitiveOperator::NotEqual => unreachable!(
                    "Inequality filter should be handled above the metadata provider level"
                ),
            };
            Ok(metadata_value_to_offset_ids
                .range::<&MetadataValue, _>(bounds)
                .map(|(_, v)| v)
                .fold(RoaringBitmap::new(), BitOr::bitor))
        } else {
            Ok(RoaringBitmap::new())
        }
    }

    pub(crate) fn search_user_ids(&self, user_ids: &[&str]) -> RoaringBitmap {
        user_ids
            .iter()
            .filter_map(|id| self.user_id_to_offset_id.get(id))
            .collect()
    }
}

pub(crate) enum MetadataProvider<'me> {
    CompactData(&'me MetadataSegmentReader<'me>),
    Log(&'me MetadataLogReader<'me>),
}

impl<'me> MetadataProvider<'me> {
    pub(crate) fn from_metadata_segment_reader(reader: &'me MetadataSegmentReader<'me>) -> Self {
        Self::CompactData(reader)
    }

    pub(crate) fn from_metadata_log_reader(reader: &'me MetadataLogReader<'me>) -> Self {
        Self::Log(reader)
    }

    pub(crate) async fn filter_by_document(
        &self,
        query: &str,
    ) -> Result<RoaringBitmap, FilterError> {
        match self {
            MetadataProvider::CompactData(metadata_segment_reader) => {
                if let Some(reader) = metadata_segment_reader.full_text_index_reader.as_ref() {
                    Ok(reader
                        .search(query)
                        .await
                        .map_err(MetadataIndexError::FullTextError)?)
                } else {
                    Ok(RoaringBitmap::new())
                }
            }
            MetadataProvider::Log(metadata_log_reader) => Ok(metadata_log_reader
                .document
                .iter()
                .filter_map(|(offset_id, document)| document.contains(query).then_some(offset_id))
                .collect()),
        }
    }

    pub(crate) async fn filter_by_metadata(
        &self,
        key: &str,
        val: &MetadataValue,
        op: &PrimitiveOperator,
    ) -> Result<RoaringBitmap, FilterError> {
        match self {
            MetadataProvider::CompactData(metadata_segment_reader) => {
                let (metadata_index_reader, kw) = match val {
                    MetadataValue::Bool(b) => (
                        metadata_segment_reader.bool_metadata_index_reader.as_ref(),
                        &(*b).into(),
                    ),
                    MetadataValue::Int(i) => (
                        metadata_segment_reader.u32_metadata_index_reader.as_ref(),
                        &(*i as u32).into(),
                    ),
                    MetadataValue::Float(f) => (
                        metadata_segment_reader.f32_metadata_index_reader.as_ref(),
                        &(*f as f32).into(),
                    ),
                    MetadataValue::Str(s) => (
                        metadata_segment_reader
                            .string_metadata_index_reader
                            .as_ref(),
                        &s.as_str().into(),
                    ),
                };
                if let Some(reader) = metadata_index_reader {
                    match op {
                        PrimitiveOperator::Equal => Ok(reader.get(key, kw).await?),
                        PrimitiveOperator::GreaterThan => Ok(reader.gt(key, kw).await?),
                        PrimitiveOperator::GreaterThanOrEqual => Ok(reader.gte(key, kw).await?),
                        PrimitiveOperator::LessThan => Ok(reader.lt(key, kw).await?),
                        PrimitiveOperator::LessThanOrEqual => Ok(reader.lte(key, kw).await?),
                        PrimitiveOperator::NotEqual => unreachable!(
                            "Inequality filter should be handled above the metadata provider level"
                        ),
                    }
                } else {
                    Ok(RoaringBitmap::new())
                }
            }
            MetadataProvider::Log(metadata_log_reader) => metadata_log_reader.get(key, val, op),
        }
    }
}

pub(crate) trait RoaringMetadataFilter<'me> {
    async fn eval(
        &'me self,
        metadata_provider: &MetadataProvider<'me>,
    ) -> Result<SignedRoaringBitmap, FilterError>;
}

impl<'me> RoaringMetadataFilter<'me> for Where {
    async fn eval(
        &'me self,
        metadata_provider: &MetadataProvider<'me>,
    ) -> Result<SignedRoaringBitmap, FilterError> {
        match self {
            Where::DirectWhereComparison(direct_comparison) => {
                direct_comparison.eval(metadata_provider).await
            }
            Where::DirectWhereDocumentComparison(direct_document_comparison) => {
                direct_document_comparison.eval(metadata_provider).await
            }
            Where::WhereChildren(where_children) => {
                // Box::pin is required to avoid infinite size future when recurse in async
                Box::pin(where_children.eval(metadata_provider)).await
            }
        }
    }
}

impl<'me> RoaringMetadataFilter<'me> for DirectWhereComparison {
    async fn eval(
        &'me self,
        metadata_provider: &MetadataProvider<'me>,
    ) -> Result<SignedRoaringBitmap, FilterError> {
        let result = match &self.comparison {
            WhereComparison::Primitive(primitive_operator, metadata_value) => {
                match primitive_operator {
                    // We convert the inequality check in to an equality check, and then negate the result
                    PrimitiveOperator::NotEqual => SignedRoaringBitmap::Exclude(
                        metadata_provider
                            .filter_by_metadata(
                                &self.key,
                                metadata_value,
                                &PrimitiveOperator::Equal,
                            )
                            .await?,
                    ),
                    PrimitiveOperator::Equal
                    | PrimitiveOperator::GreaterThan
                    | PrimitiveOperator::GreaterThanOrEqual
                    | PrimitiveOperator::LessThan
                    | PrimitiveOperator::LessThanOrEqual => SignedRoaringBitmap::Include(
                        metadata_provider
                            .filter_by_metadata(&self.key, metadata_value, primitive_operator)
                            .await?,
                    ),
                }
            }
            WhereComparison::Set(set_operator, metadata_set_value) => {
                let child_values: Vec<_> = match metadata_set_value {
                    MetadataSetValue::Bool(vec) => {
                        vec.iter().map(|b| MetadataValue::Bool(*b)).collect()
                    }
                    MetadataSetValue::Int(vec) => {
                        vec.iter().map(|i| MetadataValue::Int(*i)).collect()
                    }
                    MetadataSetValue::Float(vec) => {
                        vec.iter().map(|f| MetadataValue::Float(*f)).collect()
                    }
                    MetadataSetValue::Str(vec) => {
                        vec.iter().map(|s| MetadataValue::Str(s.clone())).collect()
                    }
                };
                let mut child_evaluations = Vec::with_capacity(child_values.len());
                for value in child_values {
                    let eval = metadata_provider
                        .filter_by_metadata(&self.key, &value, &PrimitiveOperator::Equal)
                        .await?;
                    match set_operator {
                        SetOperator::In => {
                            child_evaluations.push(SignedRoaringBitmap::Include(eval))
                        }
                        SetOperator::NotIn => {
                            child_evaluations.push(SignedRoaringBitmap::Exclude(eval))
                        }
                    };
                }
                match set_operator {
                    SetOperator::In => child_evaluations
                        .into_iter()
                        .fold(SignedRoaringBitmap::empty(), BitOr::bitor),
                    SetOperator::NotIn => child_evaluations
                        .into_iter()
                        .fold(SignedRoaringBitmap::full(), BitAnd::bitand),
                }
            }
        };
        Ok(result)
    }
}

impl<'me> RoaringMetadataFilter<'me> for DirectDocumentComparison {
    async fn eval(
        &'me self,
        metadata_provider: &MetadataProvider<'me>,
    ) -> Result<SignedRoaringBitmap, FilterError> {
        let contain = metadata_provider
            .filter_by_document(self.document.as_str())
            .await?;
        match self.operator {
            DocumentOperator::Contains => Ok(SignedRoaringBitmap::Include(contain)),
            DocumentOperator::NotContains => Ok(SignedRoaringBitmap::Exclude(contain)),
        }
    }
}

impl<'me> RoaringMetadataFilter<'me> for WhereChildren {
    async fn eval(
        &'me self,
        metadata_provider: &MetadataProvider<'me>,
    ) -> Result<SignedRoaringBitmap, FilterError> {
        let mut child_evaluations = Vec::new();
        for child in &self.children {
            child_evaluations.push(child.eval(metadata_provider).await?);
        }
        match self.operator {
            BooleanOperator::And => Ok(child_evaluations
                .into_iter()
                .fold(SignedRoaringBitmap::full(), BitAnd::bitand)),
            BooleanOperator::Or => Ok(child_evaluations
                .into_iter()
                .fold(SignedRoaringBitmap::empty(), BitOr::bitor)),
        }
    }
}

#[async_trait]
impl Operator<FilterInput, FilterOutput> for FilterOperator {
    type Error = FilterError;

    async fn run(&self, input: &FilterInput) -> Result<FilterOutput, FilterError> {
        trace!("[{}]: {:?}", self.get_name(), input);

        let record_segment_reader = input.segments.record_segment_reader().await?;
        let materializer =
            LogMaterializer::new(record_segment_reader.clone(), input.logs.clone(), None);
        let materialized_logs = materializer
            .materialize()
            .instrument(tracing::trace_span!(parent: Span::current(), "Materialize logs"))
            .await?;
        let metadata_log_reader = MetadataLogReader::new(&materialized_logs);
        let log_metadata_provider =
            MetadataProvider::from_metadata_log_reader(&metadata_log_reader);

        let metadata_segement_reader = input.segments.metadata_segment_reader().await?;
        let compact_metadata_provider =
            MetadataProvider::from_metadata_segment_reader(&metadata_segement_reader);

        // Get offset ids corresponding to user ids
        let (user_allowed_log_offset_ids, user_allowed_compact_offset_ids) =
            if let Some(user_allowed_ids) = self.query_ids.as_ref() {
                let log_offset_ids = SignedRoaringBitmap::Include(
                    metadata_log_reader.search_user_ids(
                        user_allowed_ids
                            .iter()
                            .map(String::as_str)
                            .collect::<Vec<_>>()
                            .as_slice(),
                    ),
                );
                let compact_offset_ids = if let Some(reader) = record_segment_reader.as_ref() {
                    let mut offset_ids = RoaringBitmap::new();
                    for user_id in user_allowed_ids {
                        if let Ok(offset_id) =
                            reader.get_offset_id_for_user_id(user_id.as_str()).await
                        {
                            offset_ids.insert(offset_id);
                        }
                    }
                    SignedRoaringBitmap::Include(offset_ids)
                } else {
                    SignedRoaringBitmap::full()
                };
                (log_offset_ids, compact_offset_ids)
            } else {
                (SignedRoaringBitmap::full(), SignedRoaringBitmap::full())
            };

        // Filter the offset ids in the log if the where clause is provided
        let log_offset_ids = if let Some(clause) = self.where_clause.as_ref() {
            clause.eval(&log_metadata_provider).await? & user_allowed_log_offset_ids
        } else {
            user_allowed_log_offset_ids
        };

        // Filter the offset ids in the metadata segment if the where clause is provided
        // This always exclude all offsets that is present in the materialized log
        let compact_offset_ids = if let Some(clause) = self.where_clause.as_ref() {
            clause.eval(&compact_metadata_provider).await?
                & user_allowed_compact_offset_ids
                & SignedRoaringBitmap::Exclude(metadata_log_reader.updated_offset_ids)
        } else {
            user_allowed_compact_offset_ids
                & SignedRoaringBitmap::Exclude(metadata_log_reader.updated_offset_ids)
        };

        Ok(FilterOutput {
            log_offset_ids,
            compact_offset_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use chroma_types::{
        DirectWhereComparison, MetadataValue, PrimitiveOperator, SignedRoaringBitmap, Where,
        WhereComparison,
    };

    use crate::{
        execution::{
            operator::Operator,
            operators::{fetch_segment::FetchSegmentOutput, filter::FilterOperator},
        },
        log::test::{add_delete_generator, int_as_id, LogGenerator},
        segment::test::TestSegment,
    };

    use super::FilterInput;

    async fn setup_filter_input() -> FilterInput {
        let mut test_segment = TestSegment::default();
        let generator = LogGenerator {
            generator: add_delete_generator,
        };
        test_segment.populate_with_generator(60, &generator).await;
        FilterInput {
            logs: generator.generate_chunk(61..=120),
            segments: FetchSegmentOutput {
                hnsw: test_segment.hnsw,
                blockfile: test_segment.blockfile,
                knn: test_segment.knn,
                metadata: test_segment.metadata,
                record: test_segment.record,
                collection: test_segment.collection,
            },
        }
    }

    #[tokio::test]
    async fn test_trivial() {
        let filter_input = setup_filter_input().await;

        let filter_operator = FilterOperator {
            query_ids: None,
            where_clause: None,
        };

        let filter_output = filter_operator
            .run(&filter_input)
            .await
            .expect("FilterOperator should not fail");

        assert_eq!(filter_output.log_offset_ids, SignedRoaringBitmap::full());
        assert_eq!(
            filter_output.compact_offset_ids,
            SignedRoaringBitmap::Exclude((11..=21).collect())
        );
    }

    #[tokio::test]
    async fn test_user_allowed_ids() {
        let filter_input = setup_filter_input().await;

        let filter_operator = FilterOperator {
            query_ids: Some((0..30).map(int_as_id).collect()),
            where_clause: None,
        };

        let filter_output = filter_operator
            .run(&filter_input)
            .await
            .expect("FilterOperator should not fail");

        assert_eq!(filter_output.log_offset_ids, SignedRoaringBitmap::empty());
        assert_eq!(
            filter_output.compact_offset_ids,
            SignedRoaringBitmap::Include((21..30).collect())
        );
    }

    #[tokio::test]
    async fn test_simple_eq() {
        let filter_input = setup_filter_input().await;

        let where_clause = Where::DirectWhereComparison(DirectWhereComparison {
            key: "is_even".to_string(),
            comparison: WhereComparison::Primitive(
                PrimitiveOperator::Equal,
                MetadataValue::Bool(true),
            ),
        });

        let filter_operator = FilterOperator {
            query_ids: None,
            where_clause: Some(where_clause),
        };

        let filter_output = filter_operator
            .run(&filter_input)
            .await
            .expect("FilterOperator should not fail");

        assert_eq!(
            filter_output.log_offset_ids,
            SignedRoaringBitmap::Include((51..=100).filter(|offset| offset % 2 == 0).collect())
        );
        assert_eq!(
            filter_output.compact_offset_ids,
            SignedRoaringBitmap::Include((21..=50).filter(|offset| offset % 2 == 0).collect())
        );
    }

    #[tokio::test]
    async fn test_simple_ne() {
        let filter_input = setup_filter_input().await;

        let where_clause = Where::DirectWhereComparison(DirectWhereComparison {
            key: "modulo_3".to_string(),
            comparison: WhereComparison::Primitive(
                PrimitiveOperator::NotEqual,
                MetadataValue::Int(0),
            ),
        });

        let filter_operator = FilterOperator {
            query_ids: None,
            where_clause: Some(where_clause),
        };

        let filter_output = filter_operator
            .run(&filter_input)
            .await
            .expect("FilterOperator should not fail");

        assert_eq!(
            filter_output.log_offset_ids,
            SignedRoaringBitmap::Exclude((51..=100).filter(|offset| offset % 3 == 0).collect())
        );
        assert_eq!(
            filter_output.compact_offset_ids,
            SignedRoaringBitmap::Exclude(
                (21..=50)
                    .filter(|offset| offset % 3 == 0)
                    .chain(11..=20)
                    .collect()
            )
        );
    }

    #[tokio::test]
    async fn test_simple_gt() {
        let filter_input = setup_filter_input().await;

        let where_clause = Where::DirectWhereComparison(DirectWhereComparison {
            key: "id".to_string(),
            comparison: WhereComparison::Primitive(
                PrimitiveOperator::GreaterThan,
                MetadataValue::Int(36),
            ),
        });

        let filter_operator = FilterOperator {
            query_ids: None,
            where_clause: Some(where_clause),
        };

        let filter_output = filter_operator
            .run(&filter_input)
            .await
            .expect("FilterOperator should not fail");

        assert_eq!(
            filter_output.log_offset_ids,
            SignedRoaringBitmap::Include((51..=100).collect())
        );
        assert_eq!(
            filter_output.compact_offset_ids,
            SignedRoaringBitmap::Include((37..=50).collect())
        );
    }
}
