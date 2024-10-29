from typing import Dict, Optional, Sequence, Tuple, TypedDict, Union, cast
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

import chromadb.proto.chroma_pb2 as proto
from chromadb.api.configuration import CollectionConfigurationInternal
from chromadb.api.types import Embedding, Where, WhereDocument
from chromadb.execution.expression.operator import (
    KNN,
    Filter,
    Limit,
    Projection,
    SegmentScan,
)
from chromadb.execution.expression.plan import CountPlan, GetPlan, KNNPlan
from chromadb.types import (
    Collection,
    LogRecord,
    Metadata,
    Operation,
    OperationRecord,
    RequestVersionContext,
    ScalarEncoding,
    Segment,
    SegmentScope,
    SeqId,
    UpdateMetadata,
    Vector,
    VectorEmbeddingRecord,
    VectorQueryResult,
)


class ProjectionRecord(TypedDict):
    id: str
    document: Optional[str]
    embedding: Optional[Vector]
    metadata: Optional[Metadata]


class KNNProjectionRecord(TypedDict):
    record: ProjectionRecord
    distance: Optional[float]


# TODO: Unit tests for this file, handling optional states etc
def to_proto_vector(vector: Vector, encoding: ScalarEncoding) -> proto.Vector:
    if encoding == ScalarEncoding.FLOAT32:
        as_bytes = np.array(vector, dtype=np.float32).tobytes()
        proto_encoding = proto.ScalarEncoding.FLOAT32
    elif encoding == ScalarEncoding.INT32:
        as_bytes = np.array(vector, dtype=np.int32).tobytes()
        proto_encoding = proto.ScalarEncoding.INT32
    else:
        raise ValueError(
            f"Unknown encoding {encoding}, expected one of {ScalarEncoding.FLOAT32} \
            or {ScalarEncoding.INT32}"
        )

    return proto.Vector(dimension=vector.size, vector=as_bytes, encoding=proto_encoding)


def from_proto_vector(vector: proto.Vector) -> Tuple[Embedding, ScalarEncoding]:
    encoding = vector.encoding
    as_array: Union[NDArray[np.int32], NDArray[np.float32]]
    if encoding == proto.ScalarEncoding.FLOAT32:
        as_array = np.frombuffer(vector.vector, dtype=np.float32)
        out_encoding = ScalarEncoding.FLOAT32
    elif encoding == proto.ScalarEncoding.INT32:
        as_array = np.frombuffer(vector.vector, dtype=np.int32)
        out_encoding = ScalarEncoding.INT32
    else:
        raise ValueError(
            f"Unknown encoding {encoding}, expected one of \
            {proto.ScalarEncoding.FLOAT32} or {proto.ScalarEncoding.INT32}"
        )

    return (as_array, out_encoding)


def from_proto_operation(operation: proto.Operation) -> Operation:
    if operation == proto.Operation.ADD:
        return Operation.ADD
    elif operation == proto.Operation.UPDATE:
        return Operation.UPDATE
    elif operation == proto.Operation.UPSERT:
        return Operation.UPSERT
    elif operation == proto.Operation.DELETE:
        return Operation.DELETE
    else:
        # TODO: full error
        raise RuntimeError(f"Unknown operation {operation}")


def from_proto_metadata(metadata: proto.UpdateMetadata) -> Optional[Metadata]:
    return cast(Optional[Metadata], _from_proto_metadata_handle_none(metadata, False))


def from_proto_update_metadata(
    metadata: proto.UpdateMetadata,
) -> Optional[UpdateMetadata]:
    return cast(
        Optional[UpdateMetadata], _from_proto_metadata_handle_none(metadata, True)
    )


def _from_proto_metadata_handle_none(
    metadata: proto.UpdateMetadata, is_update: bool
) -> Optional[Union[UpdateMetadata, Metadata]]:
    if not metadata.metadata:
        return None
    out_metadata: Dict[str, Union[str, int, float, bool, None]] = {}
    for key, value in metadata.metadata.items():
        if value.HasField("bool_value"):
            out_metadata[key] = value.bool_value
        elif value.HasField("string_value"):
            out_metadata[key] = value.string_value
        elif value.HasField("int_value"):
            out_metadata[key] = value.int_value
        elif value.HasField("float_value"):
            out_metadata[key] = value.float_value
        elif is_update:
            out_metadata[key] = None
        else:
            raise ValueError(f"Metadata key {key} value cannot be None")
    return out_metadata


def to_proto_update_metadata(metadata: UpdateMetadata) -> proto.UpdateMetadata:
    return proto.UpdateMetadata(
        metadata={k: to_proto_metadata_update_value(v) for k, v in metadata.items()}
    )


def from_proto_submit(
    operation_record: proto.OperationRecord, seq_id: SeqId
) -> LogRecord:
    embedding, encoding = from_proto_vector(operation_record.vector)
    record = LogRecord(
        log_offset=seq_id,
        record=OperationRecord(
            id=operation_record.id,
            embedding=embedding,
            encoding=encoding,
            metadata=from_proto_update_metadata(operation_record.metadata),
            operation=from_proto_operation(operation_record.operation),
        ),
    )
    return record


def from_proto_segment(segment: proto.Segment) -> Segment:
    return Segment(
        id=UUID(hex=segment.id),
        type=segment.type,
        scope=from_proto_segment_scope(segment.scope),
        collection=UUID(hex=segment.collection),
        metadata=from_proto_metadata(segment.metadata)
        if segment.HasField("metadata")
        else None,
    )


def to_proto_segment(segment: Segment) -> proto.Segment:
    return proto.Segment(
        id=segment["id"].hex,
        type=segment["type"],
        scope=to_proto_segment_scope(segment["scope"]),
        collection=segment["collection"].hex,
        metadata=None
        if segment["metadata"] is None
        else to_proto_update_metadata(segment["metadata"]),
    )


def from_proto_segment_scope(segment_scope: proto.SegmentScope) -> SegmentScope:
    if segment_scope == proto.SegmentScope.VECTOR:
        return SegmentScope.VECTOR
    elif segment_scope == proto.SegmentScope.METADATA:
        return SegmentScope.METADATA
    elif segment_scope == proto.SegmentScope.RECORD:
        return SegmentScope.RECORD
    else:
        raise RuntimeError(f"Unknown segment scope {segment_scope}")


def to_proto_segment_scope(segment_scope: SegmentScope) -> proto.SegmentScope:
    if segment_scope == SegmentScope.VECTOR:
        return proto.SegmentScope.VECTOR
    elif segment_scope == SegmentScope.METADATA:
        return proto.SegmentScope.METADATA
    elif segment_scope == SegmentScope.RECORD:
        return proto.SegmentScope.RECORD
    else:
        raise RuntimeError(f"Unknown segment scope {segment_scope}")


def to_proto_metadata_update_value(
    value: Union[str, int, float, bool, None]
) -> proto.UpdateMetadataValue:
    # Be careful with the order here. Since bools are a subtype of int in python,
    # isinstance(value, bool) and isinstance(value, int) both return true
    # for a value of bool type.
    if isinstance(value, bool):
        return proto.UpdateMetadataValue(bool_value=value)
    elif isinstance(value, str):
        return proto.UpdateMetadataValue(string_value=value)
    elif isinstance(value, int):
        return proto.UpdateMetadataValue(int_value=value)
    elif isinstance(value, float):
        return proto.UpdateMetadataValue(float_value=value)
    # None is used to delete the metadata key.
    elif value is None:
        return proto.UpdateMetadataValue()
    else:
        raise ValueError(
            f"Unknown metadata value type {type(value)}, expected one of str, int, \
            float, or None"
        )


def from_proto_collection(collection: proto.Collection) -> Collection:
    return Collection(
        id=UUID(hex=collection.id),
        name=collection.name,
        configuration=CollectionConfigurationInternal.from_json_str(
            collection.configuration_json_str
        ),
        metadata=from_proto_metadata(collection.metadata)
        if collection.HasField("metadata")
        else None,
        dimension=collection.dimension
        if collection.HasField("dimension") and collection.dimension
        else None,
        database=collection.database,
        tenant=collection.tenant,
        version=collection.version,
        log_position=collection.log_position,
    )


def to_proto_collection(collection: Collection) -> proto.Collection:
    return proto.Collection(
        id=collection["id"].hex,
        name=collection["name"],
        configuration_json_str=collection.get_configuration().to_json_str(),
        metadata=None
        if collection["metadata"] is None
        else to_proto_update_metadata(collection["metadata"]),
        dimension=collection["dimension"],
        tenant=collection["tenant"],
        database=collection["database"],
        version=collection["version"],
    )


def to_proto_operation(operation: Operation) -> proto.Operation:
    if operation == Operation.ADD:
        return proto.Operation.ADD
    elif operation == Operation.UPDATE:
        return proto.Operation.UPDATE
    elif operation == Operation.UPSERT:
        return proto.Operation.UPSERT
    elif operation == Operation.DELETE:
        return proto.Operation.DELETE
    else:
        raise ValueError(
            f"Unknown operation {operation}, expected one of {Operation.ADD}, \
            {Operation.UPDATE}, {Operation.UPDATE}, or {Operation.DELETE}"
        )


def to_proto_submit(
    submit_record: OperationRecord,
) -> proto.OperationRecord:
    vector = None
    if submit_record["embedding"] is not None and submit_record["encoding"] is not None:
        vector = to_proto_vector(submit_record["embedding"], submit_record["encoding"])

    metadata = None
    if submit_record["metadata"] is not None:
        metadata = to_proto_update_metadata(submit_record["metadata"])

    return proto.OperationRecord(
        id=submit_record["id"],
        vector=vector,
        metadata=metadata,
        operation=to_proto_operation(submit_record["operation"]),
    )


def from_proto_vector_embedding_record(
    embedding_record: proto.VectorEmbeddingRecord,
) -> VectorEmbeddingRecord:
    return VectorEmbeddingRecord(
        id=embedding_record.id,
        embedding=from_proto_vector(embedding_record.vector)[0],
    )


def to_proto_vector_embedding_record(
    embedding_record: VectorEmbeddingRecord,
    encoding: ScalarEncoding,
) -> proto.VectorEmbeddingRecord:
    return proto.VectorEmbeddingRecord(
        id=embedding_record["id"],
        vector=to_proto_vector(embedding_record["embedding"], encoding),
    )


def from_proto_vector_query_result(
    vector_query_result: proto.VectorQueryResult,
) -> VectorQueryResult:
    return VectorQueryResult(
        id=vector_query_result.id,
        distance=vector_query_result.distance,
        embedding=from_proto_vector(vector_query_result.vector)[0],
    )


def from_proto_request_version_context(
    request_version_context: proto.RequestVersionContext,
) -> RequestVersionContext:
    return RequestVersionContext(
        collection_version=request_version_context.collection_version,
        log_position=request_version_context.log_position,
    )


def to_proto_request_version_context(
    request_version_context: RequestVersionContext,
) -> proto.RequestVersionContext:
    return proto.RequestVersionContext(
        collection_version=request_version_context["collection_version"],
        log_position=request_version_context["log_position"],
    )


def to_proto_where(where: Where) -> proto.Where:
    response = proto.Where()
    if len(where) != 1:
        raise ValueError(f"Expected where to have exactly one operator, got {where}")

    for key, value in where.items():
        if not isinstance(key, str):
            raise ValueError(f"Expected where key to be a str, got {key}")

        if key == "$and" or key == "$or":
            if not isinstance(value, list):
                raise ValueError(
                    f"Expected where value for $and or $or to be a list of where expressions, got {value}"
                )
            children: proto.WhereChildren = proto.WhereChildren(
                children=[to_proto_where(w) for w in value]
            )
            if key == "$and":
                children.operator = proto.BooleanOperator.AND
            else:
                children.operator = proto.BooleanOperator.OR

            response.children.CopyFrom(children)
            return response

        # At this point we know we're at a direct comparison. It can either
        # be of the form {"key": "value"} or {"key": {"$operator": "value"}}.

        dc = proto.DirectComparison()
        dc.key = key

        if not isinstance(value, dict):
            # {'key': 'value'} case
            if type(value) is str:
                ssc = proto.SingleStringComparison()
                ssc.value = value
                ssc.comparator = proto.GenericComparator.EQ
                dc.single_string_operand.CopyFrom(ssc)
            elif type(value) is bool:
                sbc = proto.SingleBoolComparison()
                sbc.value = value
                sbc.comparator = proto.GenericComparator.EQ
                dc.single_bool_operand.CopyFrom(sbc)
            elif type(value) is int:
                sic = proto.SingleIntComparison()
                sic.value = value
                sic.generic_comparator = proto.GenericComparator.EQ
                dc.single_int_operand.CopyFrom(sic)
            elif type(value) is float:
                sdc = proto.SingleDoubleComparison()
                sdc.value = value
                sdc.generic_comparator = proto.GenericComparator.EQ
                dc.single_double_operand.CopyFrom(sdc)
            else:
                raise ValueError(
                    f"Expected where value to be a string, int, or float, got {value}"
                )
        else:
            for operator, operand in value.items():
                if operator in ["$in", "$nin"]:
                    if not isinstance(operand, list):
                        raise ValueError(
                            f"Expected where value for $in or $nin to be a list of values, got {value}"
                        )
                    if len(operand) == 0 or not all(
                        isinstance(x, type(operand[0])) for x in operand
                    ):
                        raise ValueError(
                            f"Expected where operand value to be a non-empty list, and all values to be of the same type "
                            f"got {operand}"
                        )
                    list_operator = None
                    if operator == "$in":
                        list_operator = proto.ListOperator.IN
                    else:
                        list_operator = proto.ListOperator.NIN
                    if type(operand[0]) is str:
                        slo = proto.StringListComparison()
                        for x in operand:
                            slo.values.extend([x])  # type: ignore
                        slo.list_operator = list_operator
                        dc.string_list_operand.CopyFrom(slo)
                    elif type(operand[0]) is bool:
                        blo = proto.BoolListComparison()
                        for x in operand:
                            blo.values.extend([x])  # type: ignore
                        blo.list_operator = list_operator
                        dc.bool_list_operand.CopyFrom(blo)
                    elif type(operand[0]) is int:
                        ilo = proto.IntListComparison()
                        for x in operand:
                            ilo.values.extend([x])  # type: ignore
                        ilo.list_operator = list_operator
                        dc.int_list_operand.CopyFrom(ilo)
                    elif type(operand[0]) is float:
                        dlo = proto.DoubleListComparison()
                        for x in operand:
                            dlo.values.extend([x])  # type: ignore
                        dlo.list_operator = list_operator
                        dc.double_list_operand.CopyFrom(dlo)
                    else:
                        raise ValueError(
                            f"Expected where operand value to be a list of strings, ints, or floats, got {operand}"
                        )
                elif operator in ["$eq", "$ne", "$gt", "$lt", "$gte", "$lte"]:
                    # Direct comparison to a single value.
                    if type(operand) is str:
                        ssc = proto.SingleStringComparison()
                        ssc.value = operand
                        if operator == "$eq":
                            ssc.comparator = proto.GenericComparator.EQ
                        elif operator == "$ne":
                            ssc.comparator = proto.GenericComparator.NE
                        else:
                            raise ValueError(
                                f"Expected where operator to be $eq or $ne, got {operator}"
                            )
                        dc.single_string_operand.CopyFrom(ssc)
                    elif type(operand) is bool:
                        sbc = proto.SingleBoolComparison()
                        sbc.value = operand
                        if operator == "$eq":
                            sbc.comparator = proto.GenericComparator.EQ
                        elif operator == "$ne":
                            sbc.comparator = proto.GenericComparator.NE
                        else:
                            raise ValueError(
                                f"Expected where operator to be $eq or $ne, got {operator}"
                            )
                        dc.single_bool_operand.CopyFrom(sbc)
                    elif type(operand) is int:
                        sic = proto.SingleIntComparison()
                        sic.value = operand
                        if operator == "$eq":
                            sic.generic_comparator = proto.GenericComparator.EQ
                        elif operator == "$ne":
                            sic.generic_comparator = proto.GenericComparator.NE
                        elif operator == "$gt":
                            sic.number_comparator = proto.NumberComparator.GT
                        elif operator == "$lt":
                            sic.number_comparator = proto.NumberComparator.LT
                        elif operator == "$gte":
                            sic.number_comparator = proto.NumberComparator.GTE
                        elif operator == "$lte":
                            sic.number_comparator = proto.NumberComparator.LTE
                        else:
                            raise ValueError(
                                f"Expected where operator to be one of $eq, $ne, $gt, $lt, $gte, $lte, got {operator}"
                            )
                        dc.single_int_operand.CopyFrom(sic)
                    elif type(operand) is float:
                        sfc = proto.SingleDoubleComparison()
                        sfc.value = operand
                        if operator == "$eq":
                            sfc.generic_comparator = proto.GenericComparator.EQ
                        elif operator == "$ne":
                            sfc.generic_comparator = proto.GenericComparator.NE
                        elif operator == "$gt":
                            sfc.number_comparator = proto.NumberComparator.GT
                        elif operator == "$lt":
                            sfc.number_comparator = proto.NumberComparator.LT
                        elif operator == "$gte":
                            sfc.number_comparator = proto.NumberComparator.GTE
                        elif operator == "$lte":
                            sfc.number_comparator = proto.NumberComparator.LTE
                        else:
                            raise ValueError(
                                f"Expected where operator to be one of $eq, $ne, $gt, $lt, $gte, $lte, got {operator}"
                            )
                        dc.single_double_operand.CopyFrom(sfc)
                    else:
                        raise ValueError(
                            f"Expected where operand value to be a string, int, or float, got {operand}"
                        )
                else:
                    # This case should never happen, as we've already
                    # handled the case for direct comparisons.
                    pass

        response.direct_comparison.CopyFrom(dc)
    return response


def to_proto_where_document(where_document: WhereDocument) -> proto.WhereDocument:
    response = proto.WhereDocument()
    if len(where_document) != 1:
        raise ValueError(
            f"Expected where_document to have exactly one operator, got {where_document}"
        )

    for operator, operand in where_document.items():
        if operator == "$and" or operator == "$or":
            # Nested "$and" or "$or" expression.
            if not isinstance(operand, list):
                raise ValueError(
                    f"Expected where_document value for $and or $or to be a list of where_document expressions, got {operand}"
                )
            children: proto.WhereDocumentChildren = proto.WhereDocumentChildren(
                children=[to_proto_where_document(w) for w in operand]
            )
            if operator == "$and":
                children.operator = proto.BooleanOperator.AND
            else:
                children.operator = proto.BooleanOperator.OR

            response.children.CopyFrom(children)
        else:
            # Direct "$contains" or "$not_contains" comparison to a single
            # value.
            if not isinstance(operand, str):
                raise ValueError(
                    f"Expected where_document operand to be a string, got {operand}"
                )
            dwd = proto.DirectWhereDocument()
            dwd.document = operand
            if operator == "$contains":
                dwd.operator = proto.WhereDocumentOperator.CONTAINS
            elif operator == "$not_contains":
                dwd.operator = proto.WhereDocumentOperator.NOT_CONTAINS
            else:
                raise ValueError(
                    f"Expected where_document operator to be one of $contains, $not_contains, got {operator}"
                )
            response.direct.CopyFrom(dwd)

    return response


def to_proto_scan(scan: SegmentScan) -> proto.ScanOperator:
    return proto.ScanOperator(
        collection=to_proto_collection(scan.collection),
        knn_id=scan.knn_id.hex,
        metadata_id=scan.metadata_id.hex,
        record_id=scan.record_id.hex,
    )


def to_proto_filter(filter: Filter) -> proto.FilterOperator:
    print(filter.user_ids)
    return proto.FilterOperator(
        ids=proto.UserIds(ids=filter.user_ids) if filter.user_ids is not None else None,
        where=to_proto_where(filter.where) if filter.where else None,
        where_document=to_proto_where_document(filter.where_document)
        if filter.where_document
        else None,
    )


def to_proto_knn(knn: KNN) -> proto.KNNOperator:
    return proto.KNNOperator(
        embeddings=[
            to_proto_vector(vector=embedding, encoding=ScalarEncoding.FLOAT32)
            for embedding in knn.embeddings
        ],
        fetch=knn.fetch,
    )


def to_proto_limit(limit: Limit) -> proto.LimitOperator:
    return proto.LimitOperator(skip=limit.skip, fetch=limit.fetch)


def to_proto_projection(projection: Projection) -> proto.ProjectionOperator:
    return proto.ProjectionOperator(
        document=projection.document,
        embedding=projection.embedding,
        metadata=projection.metadata,
    )


def to_proto_knn_projection(projection: Projection) -> proto.KNNProjectionOperator:
    return proto.KNNProjectionOperator(
        projection=to_proto_projection(projection), distance=projection.rank
    )


def to_proto_count_plan(count: CountPlan) -> proto.CountPlan:
    return proto.CountPlan(scan=to_proto_scan(count.scan))


def from_proto_count_result(result: proto.CountResult) -> int:
    return result.count


def to_proto_get_plan(get: GetPlan) -> proto.GetPlan:
    return proto.GetPlan(
        scan=to_proto_scan(get.scan),
        filter=to_proto_filter(get.filter),
        limit=to_proto_limit(get.limit),
        projection=to_proto_projection(get.projection),
    )


def from_proto_projection_record(record: proto.ProjectionRecord) -> ProjectionRecord:
    return ProjectionRecord(
        id=record.id,
        document=record.document,
        embedding=from_proto_vector(record.embedding)[0]
        if record.embedding is not None
        else None,
        metadata=from_proto_metadata(record.metadata),
    )


def from_proto_get_result(result: proto.GetResult) -> Sequence[ProjectionRecord]:
    return [from_proto_projection_record(record) for record in result.records]


def to_proto_knn_plan(knn: KNNPlan) -> proto.KNNPlan:
    return proto.KNNPlan(
        scan=to_proto_scan(knn.scan),
        filter=to_proto_filter(knn.filter),
        knn=to_proto_knn(knn.knn),
        projection=to_proto_knn_projection(knn.projection),
    )


def from_proto_knn_projection_record(
    record: proto.KNNProjectionRecord,
) -> KNNProjectionRecord:
    return KNNProjectionRecord(
        record=from_proto_projection_record(record.record), distance=record.distance
    )


def from_proto_knn_batch_result(
    results: proto.KNNBatchResult,
) -> Sequence[Sequence[KNNProjectionRecord]]:
    return [
        [from_proto_knn_projection_record(record) for record in result.records]
        for result in results.results
    ]
