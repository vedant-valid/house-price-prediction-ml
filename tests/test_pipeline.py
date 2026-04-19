from unittest.mock import patch, MagicMock


def _sample_input():
    return {
        "property_input": {
            "sqft_living": 1800, "sqft_lot": 5000, "sqft_above": 1800,
            "sqft_basement": 0, "bedrooms": 3, "bathrooms": 2.0,
            "floors": 1.0, "waterfront": 0, "view": 0, "condition": 3,
            "yr_built": 1990, "city": "Seattle", "statezip": "WA 98103",
            "year_sold": 2014, "month_sold": 6,
        },
        "predicted_price": 0.0,
        "price_range": {},
        "retrieved_docs": [],
        "retrieval_score": 0.0,
        "comparables": [],
        "report": {},
        "error": None,
    }


def test_pipeline_runs_end_to_end():
    mock_response = MagicMock()
    mock_response.content = '{"summary": "Good property.", "action": "BUY — Strong location.", "market_notes": ["Prices rising."]}'

    with patch("agent.steps.ChatGoogleGenerativeAI") as MockLLM:
        MockLLM.return_value.invoke.return_value = mock_response
        from agent.pipeline import pipeline
        result = pipeline.invoke(_sample_input())

    assert result["error"] is None
    assert result["predicted_price"] > 0
    assert "summary" in result["report"]
    assert "action" in result["report"]
    assert "disclaimer" in result["report"]


def test_pipeline_handles_missing_input():
    from agent.pipeline import pipeline
    bad_input = _sample_input()
    del bad_input["property_input"]["city"]
    result = pipeline.invoke(bad_input)
    assert result["error"] is not None
