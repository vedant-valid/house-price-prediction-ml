import os
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

MARKET_DATA_DIR = "market_data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "market_insights"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_knowledge_base(df: pd.DataFrame):
    os.makedirs(MARKET_DATA_DIR, exist_ok=True)
    _write_zip_stats(df)
    _write_city_stats(df)
    _write_feature_impact(df)
    _write_price_tiers(df)
    _write_neighborhood_rankings(df)
    _write_seasonality(df)


def _write_zip_stats(df: pd.DataFrame):
    stats = df.groupby("statezip")["price"].agg(["median", "mean", "count"])
    stats = stats[stats["count"] >= 3].sort_values("median", ascending=False)
    lines = ["ZIP Code Price Statistics for King County\n"]
    for zip_code, row in stats.iterrows():
        lines.append(
            f"{zip_code}: median ${row['median']:,.0f}, "
            f"mean ${row['mean']:,.0f}, {int(row['count'])} sales"
        )
    _save(os.path.join(MARKET_DATA_DIR, "zip_price_stats.txt"), "\n".join(lines))


def _write_city_stats(df: pd.DataFrame):
    stats = df.groupby("city")["price"].agg(["median", "mean", "count"])
    stats = stats[stats["count"] >= 3].sort_values("median", ascending=False)
    lines = ["City Price Statistics for King County\n"]
    for city, row in stats.iterrows():
        lines.append(
            f"{city}: median ${row['median']:,.0f}, "
            f"mean ${row['mean']:,.0f}, {int(row['count'])} sales"
        )
    _save(os.path.join(MARKET_DATA_DIR, "city_price_stats.txt"), "\n".join(lines))


def _write_feature_impact(df: pd.DataFrame):
    lines = ["Feature Impact on Property Prices in King County\n"]

    df2 = df.copy()
    bins = [0, 1000, 1500, 2000, 2500, 3000, 99999]
    labels = ["under 1000", "1000-1500", "1500-2000", "2000-2500", "2500-3000", "over 3000"]
    df2["sqft_bin"] = pd.cut(df2["sqft_living"], bins=bins, labels=labels)
    sqft_stats = df2.groupby("sqft_bin", observed=True)["price"].median()
    lines.append("\nMedian price by living area (sqft):")
    for label, price in sqft_stats.items():
        lines.append(f"  {label} sqft: ${price:,.0f}")

    bed_stats = df.groupby("bedrooms")["price"].median()
    lines.append("\nMedian price by bedroom count:")
    for beds, price in bed_stats.items():
        if 1 <= int(beds) <= 7:
            lines.append(f"  {int(beds)} bedrooms: ${price:,.0f}")

    if "condition" in df.columns:
        cond_stats = df.groupby("condition")["price"].median()
        lines.append("\nMedian price by condition (1=poor to 5=excellent):")
        for cond, price in cond_stats.items():
            lines.append(f"  Condition {int(cond)}: ${price:,.0f}")

    _save(os.path.join(MARKET_DATA_DIR, "feature_impact.txt"), "\n".join(lines))


def _write_price_tiers(df: pd.DataFrame):
    q1 = df["price"].quantile(0.25)
    q2 = df["price"].quantile(0.50)
    q3 = df["price"].quantile(0.75)
    q9 = df["price"].quantile(0.90)
    lines = [
        "Price Tier Analysis for King County\n",
        f"Entry-level (bottom 25%): below ${q1:,.0f}",
        f"Mid-range (25th-75th percentile): ${q1:,.0f} to ${q3:,.0f}",
        f"Median sale price: ${q2:,.0f}",
        f"Premium (top 25%): above ${q3:,.0f}",
        f"Luxury (top 10%): above ${q9:,.0f}",
    ]
    _save(os.path.join(MARKET_DATA_DIR, "price_tier_analysis.txt"), "\n".join(lines))


def _write_neighborhood_rankings(df: pd.DataFrame):
    stats = df.groupby("city")["price"].median().sort_values(ascending=False)
    lines = ["Neighborhood Price Rankings for King County\n", "Top 10 most expensive cities:"]
    for city, price in stats.head(10).items():
        lines.append(f"  {city}: ${price:,.0f}")
    lines.append("\nTop 10 most affordable cities:")
    for city, price in stats.tail(10).items():
        lines.append(f"  {city}: ${price:,.0f}")
    _save(os.path.join(MARKET_DATA_DIR, "neighborhood_rankings.txt"), "\n".join(lines))


def _write_seasonality(df: pd.DataFrame):
    lines = ["Market Seasonality for King County\n"]
    if "date" in df.columns:
        df2 = df.copy()
        df2["_month"] = pd.to_datetime(df2["date"], errors="coerce").dt.month
        month_stats = df2.groupby("_month")["price"].median()
        names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                 7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
        lines.append("Median sale price by month:")
        for m, price in month_stats.items():
            lines.append(f"  {names.get(int(m), str(m))}: ${price:,.0f}")
    else:
        lines.append("Spring (March-June) typically sees 5-10% higher sale prices.")
        lines.append("Winter months (November-February) show slightly lower prices.")
    _save(os.path.join(MARKET_DATA_DIR, "market_seasonality.txt"), "\n".join(lines))


def _save(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)


def _chunk_text(text: str, size: int = 400, overlap: int = 40) -> list:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return [c for c in chunks if c.strip()]


def build_vector_store():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    ids, texts = [], []
    for fname in sorted(os.listdir(MARKET_DATA_DIR)):
        if fname.endswith(".txt"):
            with open(os.path.join(MARKET_DATA_DIR, fname), "r") as f:
                text = f.read()
            for i, chunk in enumerate(_chunk_text(text)):
                ids.append(f"{fname}_{i}")
                texts.append(chunk)

    if not texts:
        raise ValueError(f"No .txt files found in {MARKET_DATA_DIR}. Run build_knowledge_base() first.")

    embeddings = model.encode(texts).tolist()
    collection.add(ids=ids, documents=texts, embeddings=embeddings)
    print(f"  Built vector store: {len(texts)} chunks from {len(os.listdir(MARKET_DATA_DIR))} files")
