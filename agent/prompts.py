PARSE_PROMPT = """Extract property details from the user query below and return ONLY valid JSON.

User query: "{query}"

Return JSON with these keys (use defaults if not mentioned):
- "city": string (default "Seattle")
- "statezip": string like "WA 98103" (default "WA 98103")
- "sqft_living": integer (default 1800)
- "bedrooms": integer (default 3)
- "bathrooms": float (default 2.0)
- "floors": float (default 1.0)
- "condition": integer 1-5 (default 3)
- "yr_built": integer (default 1990)
- "waterfront": integer 0 or 1 (default 0)
- "view": integer 0-4 (default 0)
- "sqft_lot": integer (default 5000)
- "sqft_above": integer (same as sqft_living if no basement mentioned)
- "sqft_basement": integer (default 0)

Respond ONLY with valid JSON, no extra text."""

REPORT_PROMPT = """You are a critical real estate investment advisor. Analyze the property below and give an honest, data-driven investment recommendation.

Property Details:
- Location: {city}, {statezip}
- Living Area: {sqft_living} sqft | Bedrooms: {bedrooms} | Bathrooms: {bathrooms}
- Condition: {condition}/5 | Year Built: {yr_built}
- Predicted Price: ${predicted_price:,.0f}
- Price Range: ${price_low:,.0f} to ${price_high:,.0f}

Market Context (from knowledge base):
{market_context}

Comparable Properties:
{comps_summary}

Decision Rules — follow these strictly:
- AVOID: condition <= 2, OR property is priced 10%+ above most comparables, OR yr_built < 1960 with poor condition
- HOLD: condition == 3 AND price is within 5% of comparables, OR market context shows uncertain/slow demand
- BUY: condition >= 4 AND price is at or below comparable median, OR strong appreciation signals in market context

Write a JSON response with exactly these three keys:
- "summary": 2-3 sentences on valuation and market position. Be honest — mention risks if present.
- "action": Start with BUY, HOLD, or AVOID. Follow with 3-4 sentences citing specific numbers from the data above.
- "market_notes": List of 2-3 short strings — each a concrete insight (risks OR opportunities), not generic praise.

Rules:
- Do NOT default to BUY. Most properties are HOLD. Only recommend BUY if the data clearly supports it.
- Cite actual numbers (price, condition score, year built, comparable prices) in your reasoning.
- Do not invent statistics not present in the context above.
- Respond ONLY with valid JSON — no markdown, no extra text."""
