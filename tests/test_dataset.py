from text2cypher_3dsg.dataset import prepare_records, split_records
from text2cypher_3dsg.records import Text2CypherRecord


def test_prepare_records_builds_chat_messages():
    records = [
        Text2CypherRecord(
            uid="0A",
            question="What objects exist?",
            gold_cypher="MATCH (o:Object) RETURN DISTINCT o.class AS class",
            query_type="class_listing",
        )
    ]
    prepared = prepare_records(records, "test_schema", "Labels in Database: Object")
    assert prepared[0].messages[0]["role"] == "system"
    assert prepared[0].messages[1]["role"] == "user"
    assert prepared[0].messages[2]["content"].startswith("MATCH")


def test_split_records_keeps_all_records():
    records = [{"uid": str(i), "query_type": "x"} for i in range(10)]
    train, eval_records, test = split_records(
        records,
        test_size=0.2,
        eval_size=0.2,
        seed=42,
        stratify_by="query_type",
    )
    assert len(train) + len(eval_records) + len(test) == 10
