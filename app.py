import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os, glob, re
from collections import Counter

st.set_page_config(page_title="Twitter Brand Sentiment", layout="wide")

# =========================
# Data Loading (robust)
# =========================
@st.cache_data
def load_data():
    # Try common names first
    candidates = [
        "data/Dataset-Train.csv",
        "data/Dataset - Train.csv",
        "data/train.csv",
        "data/Train.csv",
        "data/twitter_sentiment.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)

    # Fallback: any csv under /data
    any_csvs = glob.glob("data/**/*.csv", recursive=True)
    if any_csvs:
        st.info(f"Using detected file: {any_csvs[0]}")
        return pd.read_csv(any_csvs[0])

    st.error("CSV not found. Put your dataset inside a `data/` folder in the repo.")
    st.stop()

df = load_data()

# =========================
# Column normalization
# (edit these if your CSV headers differ)
# =========================
rename_map = {
    "tweet_text": "text",
    "emotion_in_tweet_is_directed_at": "brand",
    "is_there_an_emotion_directed_at_a_brand_or_product": "sentiment",
    # If you have a timestamp column, add it here (uncomment & adjust):
    # "tweet_created_at": "created_at",
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# Ensure key columns exist
if "text" not in df.columns:
    # Try a few common fallbacks
    for c in ["Tweet", "Text", "content", "message"]:
        if c in df.columns:
            df = df.rename(columns={c: "text"})
            break
if "sentiment" not in df.columns:
    for c in ["Sentiment", "label", "polarity"]:
        if c in df.columns:
            df = df.rename(columns={c: "sentiment"})
            break

# Minimal cleaning
df = df.dropna(subset=["text", "sentiment"])
if "brand" in df.columns:
    df["brand"] = df["brand"].fillna("Unknown")

# Optional: parse date if present
date_col = None
for c in ["created_at", "date", "timestamp", "time"]:
    if c in df.columns:
        date_col = c
        break
if date_col:
    with st.spinner("Parsing dates..."):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        # Keep only valid dates
        if df[date_col].notna().any():
            df["date_only"] = df[date_col].dt.date
        else:
            date_col = None

# =========================
# Helpers
# =========================
def top_terms(series: pd.Series, n=20):
    text = " ".join(series.astype(str)).lower()
    # keep hashtags and words (min 4 chars to cut noise)
    tokens = re.findall(r"#\w+|\b[a-z]{4,}\b", text)
    stop = {
        "https", "http", "rt", "with", "from", "this", "that", "have", "about",
        "your", "they", "them", "were", "been", "their", "there", "will", "could",
        "would", "should", "what", "when", "where", "which", "into", "because",
        "while", "after", "before", "just", "like"
    }
    tokens = [t for t in tokens if t not in stop]
    return pd.DataFrame(Counter(tokens).most_common(n), columns=["term", "freq"])

# =========================
# UI
# =========================
st.title("Twitter Brand Sentiment Dashboard")
st.caption("Actionable marketing insights from pre-labeled Twitter sentiment data")

# Sidebar filters
st.sidebar.header("Filters")

# Brand filter (if brand exists)
brand_options = ["All"]
if "brand" in df.columns:
    brand_options += sorted([b for b in df["brand"].dropna().unique()])
brand_sel = st.sidebar.selectbox("Brand", brand_options, index=0)

# Sentiment filter
sentiments = sorted(df["sentiment"].dropna().unique().tolist())
sent_sel = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)

# Date filter (if we have dates)
date_range = None
if date_col and df["date_only"].notna().any():
    min_d, max_d = df["date_only"].min(), df["date_only"].max()
    date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

# Apply filters
f = df.copy()
if brand_sel != "All" and "brand" in f.columns:
    f = f[f["brand"] == brand_sel]
if sent_sel:
    f = f[f["sentiment"].isin(sent_sel)]
if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = date_range
    if "date_only" in f.columns:
        f = f[(f["date_only"] >= start) & (f["date_only"] <= end)]

# =========================
# KPI row
# =========================
col1, col2, col3 = st.columns(3)
col1.metric("Tweets", len(f))
col2.metric("Unique brands", f["brand"].nunique() if "brand" in f.columns else 0)
pos = (f["sentiment"] == "Positive").sum()
neg = (f["sentiment"] == "Negative").sum()
neu = (f["sentiment"] == "Neutral").sum()
col3.metric("Pos / Neg / Neu", f"{pos} / {neg} / {neu}")

# =========================
# Sentiment distribution
# =========================
st.subheader("Sentiment distribution")
sent_counts = f["sentiment"].value_counts().reset_index()
sent_counts.columns = ["sentiment", "count"]
fig = px.bar(sent_counts, x="sentiment", y="count", text="count", title="Sentiment counts")
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# =========================
# Brand comparison (only if we have brand)
# =========================
if "brand" in f.columns and f["brand"].notna().sum() > 0 and brand_sel == "All":
    st.subheader("Top brands by positive %")
    tmp = f.copy()
    tmp["is_pos"] = (tmp["sentiment"] == "Positive").astype(int)
    brand_stats = (tmp.groupby("brand")["is_pos"].mean() * 100).sort_values(ascending=False)
    top = brand_stats.head(10).reset_index().rename(columns={"is_pos": "positive_pct"})
    if len(top) > 0:
        fig2 = px.bar(top, x="brand", y="positive_pct", title="Positive sentiment (%) by brand")
        fig2.update_layout(yaxis_title="Positive (%)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough brand data to compare yet.")
else:
    if "brand" not in f.columns:
        st.info("No brand column detected. Showing brand-agnostic insights below.")

# =========================
# Sentiment over
