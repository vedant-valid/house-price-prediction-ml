from agent.state import AgentState

def test_agent_state_keys():
    state: AgentState = {
        "property_input": {"sqft_living": 1800, "city": "Seattle"},
        "predicted_price": 450000.0,
        "price_range": {"low": 405000.0, "high": 495000.0},
        "retrieved_docs": [],
        "retrieval_score": 0.0,
        "comparables": [],
        "report": {},
        "error": None,
    }
    assert state["predicted_price"] == 450000.0
    assert state["error"] is None
    assert isinstance(state["comparables"], list)
