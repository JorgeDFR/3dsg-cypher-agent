from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert at translating natural language questions into Cypher for a Neo4j "
    "3D Scene Graph database. Return only the Cypher query. Do not include explanations. "
    "Do not use APOC. Use point.distance for point distances."
)


def build_user_prompt(schema: str, question: str) -> str:
    return (
        "Based on the Neo4j graph schema below, write a Cypher query that answers the "
        "question.\n\n"
        f"{schema.strip()}\n\n"
        f"Question: {question.strip()}\n\n"
        "Cypher query:"
    )


def build_messages(
    schema: str,
    question: str,
    cypher: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": build_user_prompt(schema, question)},
        {"role": "assistant", "content": cypher.strip()},
    ]
