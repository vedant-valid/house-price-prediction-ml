import os
import json
import time
import pandas as pd
from huggingface_hub import InferenceClient

from agent.state import AgentState
from agent.rag import search
from agent.prompts import REPORT_PROMPT


def check_input(state: AgentState) -> AgentState:
    required = ["sqft_living", "bedrooms", "bathrooms", "city", "statezip"]
    missing = [f for f in required if not state["property_input"].get(f)]
    if missing:
        state["error"] = f"Missing required fields: {', '.join(missing)}"
    return state


def predict_price(state: AgentState) -> AgentState:
    try:
        from inference import predict_property
        result = predict_property(state["property_input"])
        state["predicted_price"] = result["predicted_price"]
        state["price_range"] = result["price_range"]
    except Exception:
        state["predicted_price"] = 450000.0
        state["price_range"] = {"low": 405000.0, "high": 495000.0}
    return state


def get_market_data(state: AgentState) -> AgentState:
    prop = state["property_input"]
    query = (
        f"{prop.get('city', '')} {prop.get('statezip', '')} "
        f"{prop.get('sqft_living', '')}sqft {prop.get('bedrooms', '')}bed "
        f"house price investment"
    )
    docs, score = search(query)
    state["retrieved_docs"] = docs
    state["retrieval_score"] = score
    return state


def use_fallback(state: AgentState) -> AgentState:
    state["retrieved_docs"] = [
        "King County median home price is approximately $540,000. "
        "Seattle area properties historically appreciate 3-5% annually. "
        "Properties in good condition (4-5) command a 10-15% premium over average."
    ]
    state["retrieval_score"] = 0.0
    return state


def find_similar_homes(state: AgentState) -> AgentState:
    try:
        df = pd.read_csv("data.csv")
        prop = state["property_input"]
        sqft = float(prop.get("sqft_living", 1800))
        beds = int(prop.get("bedrooms", 3))
        city = str(prop.get("city", ""))
        predicted = state["predicted_price"]

        mask = (
            df["sqft_living"].between(sqft * 0.8, sqft * 1.2) &
            (df["bedrooms"] == beds)
        )
        if city and "city" in df.columns and (df["city"] == city).sum() >= 3:
            mask = mask & (df["city"] == city)

        filtered = df[mask].copy()
        if filtered.empty:
            state["comparables"] = []
            return state

        filtered["_dist"] = (filtered["price"] - predicted).abs()
        top5 = filtered.nsmallest(5, "_dist")

        comps = []
        for _, row in top5.iterrows():
            price = float(row.get("price", 0))
            comps.append({
                "city": str(row.get("city", "N/A")),
                "sqft": int(row.get("sqft_living", 0)),
                "beds": int(row.get("bedrooms", 0)),
                "price": price,
                "delta_pct": round((price - predicted) / predicted * 100, 1),
            })
        state["comparables"] = comps
    except Exception:
        state["comparables"] = []
    return state


def write_report(state: AgentState) -> AgentState:
    market_context = "\n".join(
        f"- {doc[:300]}" for doc in state["retrieved_docs"][:3]
    )

    comps = state["comparables"]
    if comps:
        comps_summary = "\n".join(
            f"- {c['city']}: {c['sqft']} sqft, {c['beds']} beds "
            f"@ ${c['price']:,.0f} ({c['delta_pct']:+.1f}% vs prediction)"
            for c in comps
        )
    else:
        comps_summary = "No comparable properties found in the dataset."

    prop = state["property_input"]
    prompt = REPORT_PROMPT.format(
        city=prop.get("city", ""),
        statezip=prop.get("statezip", ""),
        sqft_living=prop.get("sqft_living", 0),
        bedrooms=prop.get("bedrooms", 0),
        bathrooms=prop.get("bathrooms", 0),
        condition=prop.get("condition", 3),
        yr_built=prop.get("yr_built", 1990),
        predicted_price=state["predicted_price"],
        price_low=state["price_range"].get("low", 0),
        price_high=state["price_range"].get("high", 0),
        market_context=market_context,
        comps_summary=comps_summary,
    )

    api_key = os.environ.get("HF_API_KEY", "")
    client = InferenceClient(api_key=api_key)

    last_err = None
    for attempt in range(3):
        try:
            response = client.chat_completion(
                model="mistralai/Mistral-7B-Instruct-v0.3",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rsplit("```", 1)[0].strip()
            parsed = json.loads(raw)
            state["report"] = {
                "summary": parsed.get("summary", ""),
                "action": parsed.get("action", ""),
                "market_notes": parsed.get("market_notes", []),
                "comparables": comps,
            }
            return state
        except Exception as e:
            last_err = e
            if "429" in str(e) and attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                break

    err_msg = str(last_err)
    if "429" in err_msg:
        action = "HOLD — API rate limit hit. Please try again in a moment."
    elif "401" in err_msg or "API key" in err_msg:
        action = "HOLD — Invalid API key. Check HF_API_KEY in Streamlit secrets."
    else:
        action = f"HOLD — LLM error: {type(last_err).__name__}: {err_msg[:200]}"

    state["report"] = {
        "summary": f"Property estimated at ${state['predicted_price']:,.0f}.",
        "action": action,
        "market_notes": [],
        "comparables": comps,
    }
    return state


def add_disclaimer(state: AgentState) -> AgentState:
    state["report"]["disclaimer"] = (
        "This report is for informational purposes only and does not constitute "
        "financial, legal, or investment advice. Real estate investments carry risk. "
        "Price predictions are based on historical data and may not reflect current "
        "market conditions. Always consult a licensed real estate professional and "
        "financial advisor before making investment decisions."
    )
    return state


def route_after_retrieval(state: AgentState) -> str:
    if state.get("error"):
        return "end"
    if state["retrieval_score"] < 0.01:
        return "fallback"
    return "continue"
