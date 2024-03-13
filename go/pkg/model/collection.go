package model

import (
	"github.com/chroma-core/chroma/go/pkg/types"
)

type Collection struct {
	ID           types.UniqueID
	Name         string
	Topic        string
	Dimension    *int32
	Metadata     *CollectionMetadata[CollectionMetadataValueType]
	TenantID     string
	DatabaseName string
	Ts           types.Timestamp
}

type CreateCollection struct {
	ID           types.UniqueID
	Name         string
	Topic        string
	Dimension    *int32
	Metadata     *CollectionMetadata[CollectionMetadataValueType]
	GetOrCreate  bool
	TenantID     string
	DatabaseName string
	Ts           types.Timestamp
}

type DeleteCollection struct {
	ID           types.UniqueID
	TenantID     string
	DatabaseName string
	Ts           types.Timestamp
}

type UpdateCollection struct {
	ID            types.UniqueID
	Name          *string
	Topic         *string
	Dimension     *int32
	Metadata      *CollectionMetadata[CollectionMetadataValueType]
	ResetMetadata bool
	TenantID      string
	DatabaseName  string
	Ts            types.Timestamp
}

type FlushCollectionCompaction struct {
	ID                       types.UniqueID
	TenantID                 types.UniqueID
	LogPosition              int64
	CurrentCollectionVersion int32
	FlushSegmentCompactions  []*FlushSegmentCompaction
}

type FlushCollectionInfo struct {
	ID                       string
	CollectionVersion        int32
	TenantLastCompactionTime int64
}

func FilterCollection(collection *Collection, collectionID types.UniqueID, collectionName *string, collectionTopic *string) bool {
	if collectionID != types.NilUniqueID() && collectionID != collection.ID {
		return false
	}
	if collectionName != nil && *collectionName != collection.Name {
		return false
	}
	if collectionTopic != nil && *collectionTopic != collection.Topic {
		return false
	}
	return true
}
