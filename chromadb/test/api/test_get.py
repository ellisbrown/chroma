import pytest
from chromadb.errors import InvalidCollectionException
from chromadb.api.types import QueryResult
from chromadb.test.api.utils import (approx_equal, metadata_records, contains_records, logical_operator_records, records, batch_records, operator_records)


def test_get_from_db(client):
    client.reset()
    collection = client.create_collection("testspace")
    collection.add(**batch_records)
    includes = ["embeddings", "documents", "metadatas"]
    records = collection.get(include=includes)
    for key in records.keys():
        if (key in includes) or (key == "ids"):
            assert len(records[key]) == 2
        elif key == "included":
            assert set(records[key]) == set(includes)
        else:
            assert records[key] is None

def test_collection_get_with_invalid_collection_throws(client):
    client.reset()
    collection = client.create_collection("test")
    client.delete_collection("test")

    with pytest.raises(
        InvalidCollectionException, match=r"Collection .* does not exist."
    ):
        collection.get()
    

def test_get_where_document(client):
    client.reset()
    collection = client.create_collection("test_get_where_document")
    collection.add(**contains_records)

    items = collection.get(where_document={"$contains": "doc1"})
    assert len(items["metadatas"]) == 1

    items = collection.get(where_document={"$contains": "great"})
    assert len(items["metadatas"]) == 2

    items = collection.get(where_document={"$contains": "bad"})
    assert len(items["metadatas"]) == 0
    
# TEST METADATA AND METADATA FILTERING
# region
def test_where_logical_operators(client):
    client.reset()
    collection = client.create_collection("test_logical_operators")
    collection.add(**logical_operator_records)

    items = collection.get(
        where={
            "$and": [
                {"$or": [{"int_value": {"$gte": 3}}, {"float_value": {"$lt": 1.9}}]},
                {"is": "doc"},
            ]
        }
    )
    assert len(items["metadatas"]) == 3

    items = collection.get(
        where={
            "$or": [
                {
                    "$and": [
                        {"int_value": {"$eq": 3}},
                        {"string_value": {"$eq": "three"}},
                    ]
                },
                {
                    "$and": [
                        {"int_value": {"$eq": 4}},
                        {"string_value": {"$eq": "four"}},
                    ]
                },
            ]
        }
    )
    assert len(items["metadatas"]) == 2

    items = collection.get(
        where={
            "$and": [
                {
                    "$or": [
                        {"int_value": {"$eq": 1}},
                        {"string_value": {"$eq": "two"}},
                    ]
                },
                {
                    "$or": [
                        {"int_value": {"$eq": 2}},
                        {"string_value": {"$eq": "one"}},
                    ]
                },
            ]
        }
    )
    assert len(items["metadatas"]) == 2


def test_where_document_logical_operators(client):
    client.reset()
    collection = client.create_collection("test_document_logical_operators")
    collection.add(**logical_operator_records)

    items = collection.get(
        where_document={
            "$and": [
                {"$contains": "first"},
                {"$contains": "doc"},
            ]
        }
    )
    assert len(items["metadatas"]) == 1

    items = collection.get(
        where_document={
            "$or": [
                {"$contains": "first"},
                {"$contains": "second"},
            ]
        }
    )
    assert len(items["metadatas"]) == 2

    items = collection.get(
        where_document={
            "$or": [
                {"$contains": "first"},
                {"$contains": "second"},
            ]
        },
        where={
            "int_value": {"$ne": 2},
        },
    )
    assert len(items["metadatas"]) == 1

    
def test_get_include(client):
    client.reset()
    collection = client.create_collection("test_get_include")
    collection.add(**records)

    include = ["metadatas", "documents"]
    items = collection.get(include=include, where={"int_value": 1})
    assert items["embeddings"] is None
    assert items["ids"][0] == "id1"
    assert items["metadatas"][0]["int_value"] == 1
    assert items["documents"][0] == "this document is first"
    assert set(items["included"]) == set(include)

    include = ["embeddings", "documents"]
    items = collection.get(include=include)
    assert items["metadatas"] is None
    assert items["ids"][0] == "id1"
    assert approx_equal(items["embeddings"][1][0], 1.2)
    assert set(items["included"]) == set(include)

    items = collection.get(include=[])
    assert items["documents"] is None
    assert items["metadatas"] is None
    assert items["embeddings"] is None
    assert items["ids"][0] == "id1"
    assert items["included"] == []

    with pytest.raises(ValueError, match="include"):
        items = collection.get(include=["metadatas", "undefined"])

    with pytest.raises(ValueError, match="include"):
        items = collection.get(include=None)
        

# test to make sure get error on invalid id input
def test_invalid_id(client):
    client.reset()
    collection = client.create_collection("test_invalid_id")

    # Get with non-list id
    with pytest.raises(ValueError) as e:
        collection.get(ids=1)
    assert "ID" in str(e.value)

def test_get_document_valid_operators(client):
    client.reset()
    collection = client.create_collection("test_where_valid_operators")
    collection.add(**operator_records)
    with pytest.raises(ValueError, match="where document"):
        collection.get(where_document={"$lt": {"$nested": 2}})

    with pytest.raises(ValueError, match="where document"):
        collection.get(where_document={"$contains": []})

    # Test invalid $and, $or
    with pytest.raises(ValueError):
        collection.get(where_document={"$and": {"$unsupported": "doc"}})

    with pytest.raises(ValueError):
        collection.get(
            where_document={"$or": [{"$unsupported": "doc"}, {"$unsupported": "doc"}]}
        )

    with pytest.raises(ValueError):
        collection.get(where_document={"$or": [{"$contains": "doc"}]})

    with pytest.raises(ValueError):
        collection.get(where_document={"$or": []})

    with pytest.raises(ValueError):
        collection.get(
            where_document={
                "$or": [{"$and": [{"$contains": "doc"}]}, {"$contains": "doc"}]
            }
        )
        
def test_where_valid_operators(client):
    client.reset()
    collection = client.create_collection("test_where_valid_operators")
    collection.add(**operator_records)
    with pytest.raises(ValueError):
        collection.get(where={"int_value": {"$invalid": 2}})

    with pytest.raises(ValueError):
        collection.get(where={"int_value": {"$lt": "2"}})

    with pytest.raises(ValueError):
        collection.get(where={"int_value": {"$lt": 2, "$gt": 1}})

    # Test invalid $and, $or
    with pytest.raises(ValueError):
        collection.get(where={"$and": {"int_value": {"$lt": 2}}})

    with pytest.raises(ValueError):
        collection.get(
            where={"int_value": {"$lt": 2}, "$or": {"int_value": {"$gt": 1}}}
        )

    with pytest.raises(ValueError):
        collection.get(
            where={"$gt": [{"int_value": {"$lt": 2}}, {"int_value": {"$gt": 1}}]}
        )

    with pytest.raises(ValueError):
        collection.get(where={"$or": [{"int_value": {"$lt": 2}}]})

    with pytest.raises(ValueError):
        collection.get(where={"$or": []})

    with pytest.raises(ValueError):
        collection.get(where={"a": {"$contains": "test"}})

    with pytest.raises(ValueError):
        collection.get(
            where={
                "$or": [
                    {"a": {"$contains": "first"}},  # invalid
                    {"$contains": "second"},  # valid
                ]
            }
        )
        

def test_where_lt(client):
    client.reset()
    collection = client.create_collection("test_where_lt")
    collection.add(**operator_records)
    items = collection.get(where={"int_value": {"$lt": 2}})
    assert len(items["metadatas"]) == 1


def test_where_lte(client):
    client.reset()
    collection = client.create_collection("test_where_lte")
    collection.add(**operator_records)
    items = collection.get(where={"int_value": {"$lte": 2.0}})
    assert len(items["metadatas"]) == 2


def test_where_gt(client):
    client.reset()
    collection = client.create_collection("test_where_lte")
    collection.add(**operator_records)
    items = collection.get(where={"float_value": {"$gt": -1.4}})
    assert len(items["metadatas"]) == 2


def test_where_gte(client):
    client.reset()
    collection = client.create_collection("test_where_lte")
    collection.add(**operator_records)
    items = collection.get(where={"float_value": {"$gte": 2.002}})
    assert len(items["metadatas"]) == 1


def test_where_ne_string(client):
    client.reset()
    collection = client.create_collection("test_where_lte")
    collection.add(**operator_records)
    items = collection.get(where={"string_value": {"$ne": "two"}})
    assert len(items["metadatas"]) == 1


def test_where_ne_eq_number(client):
    client.reset()
    collection = client.create_collection("test_where_lte")
    collection.add(**operator_records)
    items = collection.get(where={"int_value": {"$ne": 1}})
    assert len(items["metadatas"]) == 1
    items = collection.get(where={"float_value": {"$eq": 2.002}})
    assert len(items["metadatas"]) == 1

def test_where_validation_get(client):
    client.reset()
    collection = client.create_collection("test_where_validation")
    with pytest.raises(ValueError, match="where"):
        collection.get(where={"value": {"nested": "5"}})

def test_metadata_get_where_int(client):
    client.reset()
    collection = client.create_collection("test_int")
    collection.add(**metadata_records)

    items = collection.get(where={"int_value": 1})
    assert items["metadatas"][0]["int_value"] == 1
    assert items["metadatas"][0]["string_value"] == "one"


def test_metadata_get_where_float(client):
    client.reset()
    collection = client.create_collection("test_int")
    collection.add(**metadata_records)

    items = collection.get(where={"float_value": 1.001})
    assert items["metadatas"][0]["int_value"] == 1
    assert items["metadatas"][0]["string_value"] == "one"
    assert items["metadatas"][0]["float_value"] == 1.001

def test_metadata_get_where_string(client):
    client.reset()
    collection = client.create_collection("test_int")
    collection.add(**metadata_records)

    items = collection.get(where={"string_value": "one"})
    assert items["metadatas"][0]["int_value"] == 1
    assert items["metadatas"][0]["string_value"] == "one"

def test_metadata_add_get_int_float(client):
    client.reset()
    collection = client.create_collection("test_int")
    collection.add(**metadata_records)

    items = collection.get(ids=["id1", "id2"])
    assert items["metadatas"][0]["int_value"] == 1
    assert items["metadatas"][0]["float_value"] == 1.001
    assert items["metadatas"][1]["int_value"] == 2
    assert isinstance(items["metadatas"][0]["int_value"], int)
    assert isinstance(items["metadatas"][0]["float_value"], float)


def test_metadata_add_query_int_float(client):
    client.reset()
    collection = client.create_collection("test_int")
    collection.add(**metadata_records)

    items: QueryResult = collection.query(
        query_embeddings=[[1.1, 2.3, 3.2]], n_results=1
    )
    assert items["metadatas"] is not None
    assert items["metadatas"][0][0]["int_value"] == 1
    assert items["metadatas"][0][0]["float_value"] == 1.001
    assert isinstance(items["metadatas"][0][0]["int_value"], int)
    assert isinstance(items["metadatas"][0][0]["float_value"], float)