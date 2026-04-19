from typing import TypedDict


class AgentState(TypedDict):
    property_input: dict
    predicted_price: float
    price_range: dict
    retrieved_docs: list
    retrieval_score: float
    comparables: list
    report: dict
    error: str | None
