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

import os
import pandas as pd
import tempfile

def test_rag_builder_creates_files():
    df = pd.DataFrame({
        "price": [400000, 600000, 800000, 300000, 500000],
        "sqft_living": [1200, 1800, 2500, 1000, 1600],
        "bedrooms": [2, 3, 4, 2, 3],
        "condition": [3, 4, 5, 2, 3],
        "city": ["Seattle", "Bellevue", "Mercer Island", "Auburn", "Seattle"],
        "statezip": ["WA 98103", "WA 98004", "WA 98040", "WA 98002", "WA 98115"],
    })
    import rag_builder
    orig_dir = rag_builder.MARKET_DATA_DIR
    with tempfile.TemporaryDirectory() as tmpdir:
        rag_builder.MARKET_DATA_DIR = tmpdir
        rag_builder.build_knowledge_base(df)
        rag_builder.MARKET_DATA_DIR = orig_dir
        files = os.listdir(tmpdir)
        assert "zip_price_stats.txt" in files
        assert "city_price_stats.txt" in files
        assert "feature_impact.txt" in files
        assert "price_tier_analysis.txt" in files
        assert "neighborhood_rankings.txt" in files
        assert "market_seasonality.txt" in files

def test_rag_search_returns_docs():
    from agent.rag import search
    docs, score = search("Seattle 3 bedroom house price investment", k=2)
    assert isinstance(docs, list)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert len(docs) > 0, "Expected docs from market_data/ — ensure market_data/ files exist"

from unittest.mock import patch, MagicMock

def _base_state():
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


def test_check_input_passes_valid():
    from agent.steps import check_input
    state = _base_state()
    result = check_input(state)
    assert result["error"] is None


def test_check_input_catches_missing_field():
    from agent.steps import check_input
    state = _base_state()
    del state["property_input"]["city"]
    result = check_input(state)
    assert result["error"] is not None
    assert "city" in result["error"]


def test_predict_price_sets_price():
    from agent.steps import predict_price
    state = _base_state()
    result = predict_price(state)
    assert result["predicted_price"] > 0
    assert "low" in result["price_range"]
    assert "high" in result["price_range"]


def test_use_fallback_sets_docs():
    from agent.steps import use_fallback
    state = _base_state()
    result = use_fallback(state)
    assert len(result["retrieved_docs"]) > 0
    assert result["retrieval_score"] == 0.0


def test_find_similar_homes_returns_list():
    from agent.steps import find_similar_homes
    state = _base_state()
    state["predicted_price"] = 450000.0
    result = find_similar_homes(state)
    assert isinstance(result["comparables"], list)


def test_add_disclaimer_appends_text():
    from agent.steps import add_disclaimer
    state = _base_state()
    state["report"] = {"summary": "test", "action": "BUY", "market_notes": []}
    result = add_disclaimer(state)
    assert "disclaimer" in result["report"]
    assert len(result["report"]["disclaimer"]) > 20


def test_write_report_fallback_on_bad_response():
    from agent.steps import write_report
    state = _base_state()
    state["predicted_price"] = 450000.0
    state["price_range"] = {"low": 405000.0, "high": 495000.0}
    state["retrieved_docs"] = ["King County median price is $540,000."]
    state["comparables"] = []

    mock_response = MagicMock()
    mock_response.content = "this is not json {{{broken"

    with patch("agent.steps.ChatGoogleGenerativeAI") as MockLLM:
        MockLLM.return_value.invoke.return_value = mock_response
        result = write_report(state)

    assert "summary" in result["report"]
    assert "action" in result["report"]
