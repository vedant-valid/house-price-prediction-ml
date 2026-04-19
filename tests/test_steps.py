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
