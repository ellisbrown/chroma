use std::iter::once;
use std::str::FromStr;

use crate::config::QueryServiceConfig;
use crate::execution::dispatcher::Dispatcher;
use crate::execution::operators::fetch_log::FetchLogOperator;
use crate::execution::operators::fetch_segment::FetchSegmentOperator;
use crate::execution::operators::knn_projection::KnnProjectionOperator;
use crate::execution::orchestration::get::GetOrchestrator;
use crate::execution::orchestration::knn::{KnnError, KnnFilterOrchestrator, KnnOrchestrator};
use crate::execution::orchestration::CountQueryOrchestrator;
use crate::log::log::Log;
use crate::sysdb::sysdb::SysDb;
use crate::system::{ComponentHandle, System};
use crate::tracing::util::wrap_span_with_parent_context;
use crate::utils::convert::{from_proto_knn, to_proto_knn_batch_result};
use async_trait::async_trait;
use chroma_blockstore::provider::BlockfileProvider;
use chroma_config::Configurable;
use chroma_error::ChromaError;
use chroma_index::hnsw_provider::HnswIndexProvider;
use chroma_types::chroma_proto::query_executor_server::QueryExecutor;
use chroma_types::chroma_proto::{
    self, CountPlan, CountResult, GetPlan, GetResult, KnnBatchResult, KnnPlan,
};
use chroma_types::CollectionUuid;
use futures::future::try_join_all;
use tokio::signal::unix::{signal, SignalKind};
use tonic::{transport::Server, Request, Response, Status};
use tracing::{trace_span, Instrument};
use uuid::Uuid;

#[derive(Clone)]
pub struct WorkerServer {
    // System
    system: Option<System>,
    // Component dependencies
    dispatcher: Option<ComponentHandle<Dispatcher>>,
    // Service dependencies
    log: Box<Log>,
    sysdb: Box<SysDb>,
    hnsw_index_provider: HnswIndexProvider,
    blockfile_provider: BlockfileProvider,
    port: u16,
}

#[async_trait]
impl Configurable<QueryServiceConfig> for WorkerServer {
    async fn try_from_config(config: &QueryServiceConfig) -> Result<Self, Box<dyn ChromaError>> {
        let sysdb_config = &config.sysdb;
        let sysdb = match crate::sysdb::from_config(sysdb_config).await {
            Ok(sysdb) => sysdb,
            Err(err) => {
                tracing::error!("Failed to create sysdb component: {:?}", err);
                return Err(err);
            }
        };
        let log_config = &config.log;
        let log = match crate::log::from_config(log_config).await {
            Ok(log) => log,
            Err(err) => {
                tracing::error!("Failed to create log component: {:?}", err);
                return Err(err);
            }
        };
        let storage = match chroma_storage::from_config(&config.storage).await {
            Ok(storage) => storage,
            Err(err) => {
                tracing::error!("Failed to create storage component: {:?}", err);
                return Err(err);
            }
        };

        let blockfile_provider = BlockfileProvider::try_from_config(&(
            config.blockfile_provider.clone(),
            storage.clone(),
        ))
        .await?;
        let hnsw_index_provider =
            HnswIndexProvider::try_from_config(&(config.hnsw_provider.clone(), storage.clone()))
                .await?;
        Ok(WorkerServer {
            dispatcher: None,
            system: None,
            sysdb,
            log,
            hnsw_index_provider,
            blockfile_provider,
            port: config.my_port,
        })
    }
}

impl WorkerServer {
    pub(crate) async fn run(worker: WorkerServer) -> Result<(), Box<dyn std::error::Error>> {
        let addr = format!("[::]:{}", worker.port).parse().unwrap();
        println!("Worker listening on {}", addr);
        let server = Server::builder().add_service(
            chroma_proto::query_executor_server::QueryExecutorServer::new(worker.clone()),
        );

        #[cfg(debug_assertions)]
        let server =
            server.add_service(chroma_proto::debug_server::DebugServer::new(worker.clone()));

        let server = server.serve_with_shutdown(addr, async {
            let mut sigterm = match signal(SignalKind::terminate()) {
                Ok(sigterm) => sigterm,
                Err(e) => {
                    tracing::error!("Failed to create signal handler: {:?}", e);
                    return;
                }
            };
            sigterm.recv().await;
            tracing::info!("Received SIGTERM, shutting down");
        });

        server.await?;
        Ok(())
    }

    pub(crate) fn set_dispatcher(&mut self, dispatcher: ComponentHandle<Dispatcher>) {
        self.dispatcher = Some(dispatcher);
    }

    pub(crate) fn set_system(&mut self, system: System) {
        self.system = Some(system);
    }

    fn decompose_proto_scan(
        &self,
        scan: chroma_proto::ScanOperator,
    ) -> Result<(FetchLogOperator, FetchSegmentOperator), Status> {
        let collection = scan
            .collection
            .ok_or(Status::invalid_argument("Invalid Collection"))?;
        let collection_uuid = CollectionUuid::from_str(&collection.id)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        let knn_uuid = Uuid::parse_str(&collection.id)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        let metadata_uuid = Uuid::parse_str(&collection.id)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        let record_uuid = Uuid::parse_str(&collection.id)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        Ok((
            FetchLogOperator {
                log_client: self.log.clone(),
                // TODO: Make this configurable
                batch_size: 100,
                start_log_offset_id: collection.log_position as u32 + 1,
                maximum_fetch_count: None,
                collection_uuid,
            },
            FetchSegmentOperator {
                sysdb: self.sysdb.clone(),
                collection_uuid,
                collection_version: collection.version as u32,
                metadata_uuid,
                record_uuid,
                vector_uuid: knn_uuid,
            },
        ))
    }

    async fn orchestrate_count(
        &self,
        count: Request<CountPlan>,
    ) -> Result<Response<CountResult>, Status> {
        let dispatcher = self
            .dispatcher
            .as_ref()
            .ok_or(Status::internal("Dispatcher is not initialized"))?;
        let system = self
            .system
            .as_ref()
            .ok_or(Status::internal("System is not initialized"))?;

        let scan = count
            .into_inner()
            .scan
            .ok_or(Status::invalid_argument("Invalid Scan Operator"))?;
        let (fetch_log_operator, fetch_segment_operator) = self.decompose_proto_scan(scan)?;

        // TODO: Update Count Orchestractor
        let counter = CountQueryOrchestrator::new(
            system.clone(),
            &fetch_segment_operator.metadata_uuid,
            &fetch_log_operator.collection_uuid,
            fetch_log_operator.log_client,
            fetch_segment_operator.sysdb,
            dispatcher.clone(),
            self.blockfile_provider.clone(),
            fetch_segment_operator.collection_version,
            fetch_log_operator.start_log_offset_id as u64,
        );

        match counter.run().await {
            Ok(count) => Ok(Response::new(CountResult {
                count: count as u32,
            })),
            Err(err) => Err(Status::new(err.code().into(), err.to_string())),
        }
    }

    async fn orchestrate_get(&self, get: Request<GetPlan>) -> Result<Response<GetResult>, Status> {
        let dispatcher = self
            .dispatcher
            .as_ref()
            .ok_or(Status::internal("Dispatcher is not initialized"))?;
        let system = self
            .system
            .as_ref()
            .ok_or(Status::internal("System is not initialized"))?;

        let get_inner = get.into_inner();
        let scan = get_inner
            .scan
            .ok_or(Status::invalid_argument("Invalid Scan Operator"))?;
        let (fetch_log_operator, fetch_segment_operator) = self.decompose_proto_scan(scan)?;
        let filter = get_inner
            .filter
            .ok_or(Status::invalid_argument("Invalid Filter Operator"))?;
        let limit = get_inner
            .limit
            .ok_or(Status::invalid_argument("Invalid Scan Operator"))?;
        let projection = get_inner
            .projection
            .ok_or(Status::invalid_argument("Invalid Projection Operator"))?;

        let getter = GetOrchestrator::new(
            self.blockfile_provider.clone(),
            dispatcher.clone(),
            // TODO: Make this configurable
            1000,
            fetch_log_operator,
            fetch_segment_operator,
            filter.try_into()?,
            limit.into(),
            projection.into(),
        );

        match getter.run(system.clone()).await {
            Ok(result) => Ok(Response::new(result.try_into()?)),
            Err(err) => Err(Status::new(err.code().into(), err.to_string())),
        }
    }

    async fn orchestrate_knn(
        &self,
        knn: Request<KnnPlan>,
    ) -> Result<Response<KnnBatchResult>, Status> {
        let dispatcher = self
            .dispatcher
            .as_ref()
            .ok_or(Status::internal("Dispatcher is not initialized"))?;
        let system = self
            .system
            .as_ref()
            .ok_or(Status::internal("System is not initialized"))?;

        let knn_inner = knn.into_inner();
        let scan = knn_inner
            .scan
            .ok_or(Status::invalid_argument("Invalid Scan Operator"))?;
        let (fetch_log_operator, fetch_segment_operator) = self.decompose_proto_scan(scan)?;
        let filter = knn_inner
            .filter
            .ok_or(Status::invalid_argument("Invalid Filter Operator"))?;
        let knn = knn_inner
            .knn
            .ok_or(Status::invalid_argument("Invalid Scan Operator"))?;

        let sieve = KnnFilterOrchestrator::new(
            self.blockfile_provider.clone(),
            dispatcher.clone(),
            // TODO: Make this configurable
            1000,
            fetch_log_operator,
            fetch_segment_operator,
            filter.try_into()?,
        );

        let matching_records = match sieve.run(system.clone()).await {
            Ok(output) => output,
            Err(KnnError::EmptyCollection) => {
                return Ok(Response::new(to_proto_knn_batch_result(
                    once(Default::default())
                        .cycle()
                        .take(knn.embeddings.len())
                        .collect(),
                )?));
            }
            Err(e) => {
                return Err(Status::new(e.code().into(), e.to_string()));
            }
        };

        let projection = knn_inner
            .projection
            .ok_or(Status::invalid_argument("Invalid Projection Operator"))?;
        let knn_projection = KnnProjectionOperator::try_from(projection)?;

        match try_join_all(
            from_proto_knn(knn)?
                .into_iter()
                .map(|knn| {
                    KnnOrchestrator::new(
                        self.blockfile_provider.clone(),
                        dispatcher.clone(),
                        self.hnsw_index_provider.clone(),
                        // TODO: Make this configurable
                        1000,
                        matching_records.clone(),
                        knn,
                        knn_projection.clone(),
                    )
                })
                .map(|knner| knner.run(system.clone())),
        )
        .await
        {
            Ok(results) => Ok(Response::new(to_proto_knn_batch_result(results)?)),
            Err(err) => Err(Status::new(err.code().into(), err.to_string())),
        }
    }
}

#[tonic::async_trait]
impl QueryExecutor for WorkerServer {
    async fn count(&self, count: Request<CountPlan>) -> Result<Response<CountResult>, Status> {
        // Note: We cannot write a middleware that instruments every service rpc
        // with a span because of https://github.com/hyperium/tonic/pull/1202.
        let count_span = trace_span!(
            "CountPlan",
            count = ?count
        );
        let instrumented_span = wrap_span_with_parent_context(count_span, count.metadata());
        self.orchestrate_count(count)
            .instrument(instrumented_span)
            .await
    }

    async fn get(&self, get: Request<GetPlan>) -> Result<Response<GetResult>, Status> {
        // Note: We cannot write a middleware that instruments every service rpc
        // with a span because of https://github.com/hyperium/tonic/pull/1202.
        let get_span = trace_span!(
            "GetPlan",
            get = ?get
        );
        let instrumented_span = wrap_span_with_parent_context(get_span, get.metadata());
        self.orchestrate_get(get)
            .instrument(instrumented_span)
            .await
    }

    async fn knn(&self, knn: Request<KnnPlan>) -> Result<Response<KnnBatchResult>, Status> {
        // Note: We cannot write a middleware that instruments every service rpc
        // with a span because of https://github.com/hyperium/tonic/pull/1202.
        let knn_span = trace_span!(
            "KnnPlan",
            knn = ?knn
        );
        let instrumented_span = wrap_span_with_parent_context(knn_span, knn.metadata());
        self.orchestrate_knn(knn)
            .instrument(instrumented_span)
            .await
    }
}

#[cfg(debug_assertions)]
#[tonic::async_trait]
impl chroma_proto::debug_server::Debug for WorkerServer {
    async fn get_info(
        &self,
        request: Request<()>,
    ) -> Result<Response<chroma_proto::GetInfoResponse>, Status> {
        // Note: We cannot write a middleware that instruments every service rpc
        // with a span because of https://github.com/hyperium/tonic/pull/1202.
        let request_span = trace_span!("Get info");

        wrap_span_with_parent_context(request_span, request.metadata()).in_scope(|| {
            let response = chroma_proto::GetInfoResponse {
                version: option_env!("CARGO_PKG_VERSION")
                    .unwrap_or("unknown")
                    .to_string(),
            };
            Ok(Response::new(response))
        })
    }

    async fn trigger_panic(&self, request: Request<()>) -> Result<Response<()>, Status> {
        // Note: We cannot write a middleware that instruments every service rpc
        // with a span because of https://github.com/hyperium/tonic/pull/1202.
        let request_span = trace_span!("Trigger panic");

        wrap_span_with_parent_context(request_span, request.metadata()).in_scope(|| {
            panic!("Intentional panic triggered");
        })
    }
}

#[cfg(test)]
mod tests {
    #[cfg(debug_assertions)]
    use super::*;
    #[cfg(debug_assertions)]
    use crate::execution::dispatcher;
    #[cfg(debug_assertions)]
    use crate::log::log::InMemoryLog;
    #[cfg(debug_assertions)]
    use crate::sysdb::test_sysdb::TestSysDb;
    #[cfg(debug_assertions)]
    use crate::system;
    #[cfg(debug_assertions)]
    use chroma_blockstore::arrow::config::TEST_MAX_BLOCK_SIZE_BYTES;
    #[cfg(debug_assertions)]
    use chroma_cache::{new_cache_for_test, new_non_persistent_cache_for_test};
    #[cfg(debug_assertions)]
    use chroma_proto::debug_client::DebugClient;
    #[cfg(debug_assertions)]
    use chroma_storage::{local::LocalStorage, Storage};
    #[cfg(debug_assertions)]
    use tempfile::tempdir;

    #[tokio::test]
    #[cfg(debug_assertions)]
    async fn gracefully_handles_panics() {
        let sysdb = TestSysDb::new();
        let log = InMemoryLog::new();
        let tmp_dir = tempdir().unwrap();
        let storage = Storage::Local(LocalStorage::new(tmp_dir.path().to_str().unwrap()));
        let block_cache = new_cache_for_test();
        let sparse_index_cache = new_cache_for_test();
        let hnsw_index_cache = new_non_persistent_cache_for_test();
        let (_, rx) = tokio::sync::mpsc::unbounded_channel();
        let port = random_port::PortPicker::new().pick().unwrap();
        let mut server = WorkerServer {
            dispatcher: None,
            system: None,
            sysdb: Box::new(SysDb::Test(sysdb)),
            log: Box::new(Log::InMemory(log)),
            hnsw_index_provider: HnswIndexProvider::new(
                storage.clone(),
                tmp_dir.path().to_path_buf(),
                hnsw_index_cache,
                rx,
            ),
            blockfile_provider: BlockfileProvider::new_arrow(
                storage,
                TEST_MAX_BLOCK_SIZE_BYTES,
                block_cache,
                sparse_index_cache,
            ),
            port,
        };

        let system: system::System = system::System::new();
        let dispatcher = dispatcher::Dispatcher::new(4, 10, 10);
        let dispatcher_handle = system.start_component(dispatcher);

        server.set_system(system.clone());
        server.set_dispatcher(dispatcher_handle);

        tokio::spawn(async move {
            let _ = crate::server::WorkerServer::run(server).await;
        });

        let mut client = DebugClient::connect(format!("http://localhost:{}", port))
            .await
            .unwrap();

        // Test response when handler panics
        let err_response = client.trigger_panic(Request::new(())).await.unwrap_err();
        assert_eq!(err_response.code(), tonic::Code::Cancelled);

        // The server should still work, even after a panic was thrown
        let response = client.get_info(Request::new(())).await;
        assert!(response.is_ok());
    }
}
