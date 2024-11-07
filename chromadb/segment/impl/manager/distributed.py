from collections import defaultdict
from threading import Lock
from typing import Dict, Optional, Sequence, Type
from uuid import UUID, uuid4

from overrides import override

from chromadb.config import System, get_class
from chromadb.db.system import SysDB
from chromadb.segment import (
    SegmentImplementation,
    SegmentManager,
    SegmentType,
)
from chromadb.segment.distributed import SegmentDirectory
from chromadb.telemetry.opentelemetry import (
    OpenTelemetryClient,
    OpenTelemetryGranularity,
    trace_method,
)
from chromadb.types import Collection, Metadata, Operation, Segment, SegmentScope

# TODO: it is odd that the segment manager is different for distributed vs local
# implementations.  This should be refactored to be more consistent and shared.
# needed in this is the ability to specify the desired segment types for a collection
# It is odd that segment manager is coupled to the segment implementation. We need to rethink
# this abstraction.

SEGMENT_TYPE_IMPLS = {
    SegmentType.SQLITE: "chromadb.segment.impl.metadata.sqlite.SqliteMetadataSegment",
    SegmentType.HNSW_DISTRIBUTED: "chromadb.segment.impl.vector.grpc_segment.GrpcVectorSegment",
    SegmentType.BLOCKFILE_METADATA: "chromadb.segment.impl.metadata.grpc_segment.GrpcMetadataSegment",
}


class DistributedSegmentManager(SegmentManager):
    _sysdb: SysDB
    _system: System
    _opentelemetry_client: OpenTelemetryClient
    _instances: Dict[UUID, SegmentImplementation]
    _segment_cache: Dict[
        UUID, Dict[SegmentScope, Segment]
    ]  # collection_id -> scope -> segment
    _segment_directory: SegmentDirectory
    _lock: Lock
    # _segment_server_stubs: Dict[str, SegmentServerStub]  # grpc_url -> grpc stub

    def __init__(self, system: System):
        super().__init__(system)
        self._sysdb = self.require(SysDB)
        self._segment_directory = self.require(SegmentDirectory)
        self._system = system
        self._opentelemetry_client = system.require(OpenTelemetryClient)
        self._instances = {}
        self._segment_cache = defaultdict(dict)
        self._lock = Lock()

    @trace_method(
        "DistributedSegmentManager.create_segments",
        OpenTelemetryGranularity.OPERATION_AND_SEGMENT,
    )
    @override
    def create_segments(self, collection: Collection) -> Sequence[Segment]:
        vector_segment = _segment(
            SegmentType.HNSW_DISTRIBUTED, SegmentScope.VECTOR, collection
        )
        metadata_segment = _segment(
            SegmentType.BLOCKFILE_METADATA, SegmentScope.METADATA, collection
        )
        record_segment = _segment(
            SegmentType.BLOCKFILE_RECORD, SegmentScope.RECORD, collection
        )
        return [vector_segment, record_segment, metadata_segment]

    @override
    def delete_segments(self, collection_id: UUID) -> Sequence[UUID]:
        segments = self._sysdb.get_segments(collection=collection_id)
        return [s["id"] for s in segments]

    @trace_method(
        "DistributedSegmentManager.get_segment",
        OpenTelemetryGranularity.OPERATION_AND_SEGMENT,
    )
    def get_segment(self, collection_id: UUID, scope: SegmentScope) -> Segment:
        if scope not in self._segment_cache[collection_id]:
            segments = self._sysdb.get_segments(collection=collection_id, scope=scope)
            known_types = set([k.value for k in SEGMENT_TYPE_IMPLS.keys()])
            # Get the first segment of a known type
            segment = next(filter(lambda s: s["type"] in known_types, segments))
            # TODO: Register a callback to update the segment when it gets moved
            # self._segment_directory.register_updated_segment_callback()
            self._segment_cache[collection_id][scope] = segment
        return self._segment_cache[collection_id][scope]

    @trace_method(
        "DistributedSegmentManager.get_endpoint",
        OpenTelemetryGranularity.OPERATION_AND_SEGMENT,
    )
    def get_endpoint(self, collection_id: UUID) -> str:
        # Get grpc endpoint from record segment. Since grpc endpoint is endpoint is
        # determined by collection uuid, the endpoint should be the same for all
        # segments of the same collection
        record_segment = self.get_segment(collection_id, SegmentScope.RECORD)
        return self._segment_directory.get_segment_endpoint(record_segment)

    @trace_method(
        "DistributedSegmentManager.hint_use_collection",
        OpenTelemetryGranularity.OPERATION_AND_SEGMENT,
    )
    @override
    def hint_use_collection(self, collection_id: UUID, hint_type: Operation) -> None:
        pass

    # TODO: rethink duplication from local segment manager
    def _cls(self, segment: Segment) -> Type[SegmentImplementation]:
        classname = SEGMENT_TYPE_IMPLS[SegmentType(segment["type"])]
        cls = get_class(classname, SegmentImplementation)
        return cls

    def _instance(self, segment: Segment) -> SegmentImplementation:
        if segment["id"] not in self._instances:
            cls = self._cls(segment)
            instance = cls(self._system, segment)
            instance.start()
            self._instances[segment["id"]] = instance
        return self._instances[segment["id"]]


# TODO: rethink duplication from local segment manager
def _segment(type: SegmentType, scope: SegmentScope, collection: Collection) -> Segment:
    """Create a metadata dict, propagating metadata correctly for the given segment type."""

    metadata: Optional[Metadata] = None
    # For the segment types with python implementations, we can propagate metadata
    if type in SEGMENT_TYPE_IMPLS:
        cls = get_class(SEGMENT_TYPE_IMPLS[type], SegmentImplementation)
        collection_metadata = collection.metadata
        if collection_metadata:
            metadata = cls.propagate_collection_metadata(collection_metadata)

    return Segment(
        id=uuid4(),
        type=type.value,
        scope=scope,
        collection=collection.id,
        metadata=metadata,
    )
