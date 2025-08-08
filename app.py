# app.py — Social Media Brand Sentiment (Beginner-friendly, Actionable)
import os, re
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="Social Media Sentiment BI", layout="wide")
st.title("📊 Social Media Brand Sentiment — BI Dashboard")

# -------------------
# Sentiment model (VADER)
# -------------------
@st.cache_resource
def load_vader():
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    from nltk.sentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

def predict_sentiment(texts):
    sia = load_vader()
    labels = []
    for t in texts:
        c = sia.polarity_scores(str(t))["compound"]
        labels.append("Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral"))
    return labels

# -------------------
# Data helpers
# -------------------
os.makedirs("data", exist_ok=True)

def read_csv_or_upload(label, expected_path):
    st.subheader(f"{label} data")
    st.caption(f"Looking for **{expected_path}**")
    if os.path.exists(expected_path):
        st.success(f"Found: {expected_path}")
        return pd.read_csv(expected_path)
    up = st.file_uploader(f"Upload {label} CSV", type=["csv"], key=label)
    if up is not None:
        tmp_path = f"data/{label.lower()}.csv"
        with open(tmp_path, "wb") as f:
            f.write(up.read())
        st.success(f"Saved to {tmp_path}. Click ▶ Rerun at the top.")
        st.stop()
    return None

def rename_basic_cols(df):
    # Make sure we at least have `text`; map common names
    mapping = {"tweet_text": "text", "Tweet": "text", "Text": "text", "text_clean": "text"}
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    # Optional brand column name commonly seen in some datasets
    if "emotion_in_tweet_is_directed_at" in df.columns and "brand" not in df.columns:
        df = df.rename(columns={"emotion_in_tweet_is_directed_at": "brand"})
    return df

def parse_date_if_present(df):
    # If you have a date column, rename to created_at and parse; else skip
    for cand in ["created_at", "tweet_created_at", "date", "timestamp", "time"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "created_at"})
            break
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        if df["created_at"].notna().any():
            df["day"] = df["created_at"].dt.date
    return df

# -------------------
# Load train / test
# -------------------
train_df = read_csv_or_upload("Train", "data/train.csv")
test_df  = read_csv_or_upload("Test (optional)", "data/test.csv")

if train_df is None:
    st.stop()

train_df = rename_basic_cols(train_df)
test_df  = rename_basic_cols(test_df) if test_df is not None else None

if "text" not in train_df.columns:
    st.error("Your TRAIN file must have a text column (e.g., 'tweet_text' or 'text').")
    st.stop()

# clean brand if present
if "brand" in train_df.columns:
    train_df["brand"] = train_df["brand"].fillna("Unknown")
if test_df is not None and "brand" in test_df.columns:
    test_df["brand"] = test_df["brand"].fillna("Unknown")

# parse any available date/time for trend
train_df = parse_date_if_present(train_df)
if test_df is not None:
    test_df = parse_date_if_present(test_df)

st.divider()

# -------------------
# Quick model check (only if train has labels)
# -------------------
st.header("✅ Quick Model Check (train split)")
if "sentiment" in train_df.columns and train_df["sentiment"].notna().any():
    labeled = train_df.dropna(subset=["text", "sentiment"])
    if len(labeled) > 20:
        tr, val = train_test_split(labeled, test_size=0.2, random_state=42)
        y_true = val["sentiment"].astype(str)
        y_pred = predict_sentiment(val["text"].tolist())
        st.text("Classification report:")
        st.text(classification_report(y_true, y_pred))
    else:
        st.info("Not enough labeled rows to evaluate. Skipping.")
else:
    st.info("No 'sentiment' column in train. Skipping evaluation (we'll predict sentiments for BI).")

st.divider()

# -------------------
# Predict sentiments where missing
# -------------------
if "sentiment" not in train_df.columns or train_df["sentiment"].isna().all():
    train_df["sentiment"] = predict_sentiment(train_df["text"].tolist())

if test_df is not None:
    if "text" not in test_df.columns:
        st.warning("Test file has no 'text' column; it will be ignored.")
        test_df = None
    elif "sentiment" not in test_df.columns or test_df["sentiment"].isna().all():
        test_df["sentiment"] = predict_sentiment(test_df["text"].tolist())

train_df["source"] = "Train"
all_df = pd.concat([train_df, test_df.assign(source="Test")] , ignore_index=True) if test_df is not None else train_df.copy()

# -------------------
# BI: Core charts
# -------------------
st.header("📊 Core Insights")

# Overall sentiment (pie)
sent_counts = all_df["sentiment"].astype(str).value_counts().reset_index()
sent_counts.columns = ["Sentiment", "Count"]
fig_pie = px.pie(sent_counts, names="Sentiment", values="Count", title="Overall Sentiment")
st.plotly_chart(fig_pie, use_container_width=True)

# Top-5 brands sentiment split (stacked bar)
if "brand" in all_df.columns:
    st.subheader("Top 5 brands — sentiment split")
    top5 = all_df["brand"].value_counts().head(5).index
    small = all_df[all_df["brand"].isin(top5)]
    brand_sent = small.groupby(["brand","sentiment"]).size().reset_index(name="count")
    fig_bar = px.bar(brand_sent, x="brand", y="count", color="sentiment", barmode="stack",
                     title="Sentiment by brand (Top 5)")
    st.plotly_chart(fig_bar, use_container_width=True)

# Share of Positive by brand (bar)
if "brand" in all_df.columns:
    st.subheader("Share of Positive (%) by brand")
    g = all_df.groupby(["brand","sentiment"]).size().reset_index(name="n")
    pivot = g.pivot(index="brand", columns="sentiment", values="n").fillna(0)
    pivot["SoP"] = (pivot.get("Positive", 0) / pivot.sum(axis=1) * 100).round(1)
    sop = pivot["SoP"].sort_values(ascending=False).head(10).reset_index()
    fig_sop = px.bar(sop, x="brand", y="SoP", title="Top brands by Share of Positive (SoP)")
    st.plotly_chart(fig_sop, use_container_width=True)

# Trend over time (if date present)
if "day" in all_df.columns and all_df["day"].notna().any():
    st.subheader("Sentiment trend over time")
    trend = all_df.dropna(subset=["day"]).groupby(["day","sentiment"]).size().reset_index(name="count")
    fig_trend = px.line(trend, x="day", y="count", color="sentiment", markers=True, title="Daily sentiment counts")
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No date/timestamp column detected — add one to see trends.")

st.divider()

# -------------------
# BI: Drivers (what to fix / what to amplify)
# -------------------
st.header("🔎 Drivers & Guidance")

# Negative themes via bigrams
def top_bigrams(texts, n=12):
    pairs = []
    for t in texts:
        words = re.findall(r"[a-z]{3,}", str(t).lower())
        pairs += list(zip(words, words[1:]))
    counts = Counter(pairs).most_common(n)
    return pd.DataFrame([(" ".join(k), v) for k, v in counts], columns=["bigram","count"])

neg_issues = top_bigrams(all_df[all_df["sentiment"]=="Negative"]["text"]) if "text" in all_df.columns else pd.DataFrame(columns=["bigram","count"])
st.subheader("Top negative themes (bigrams)")
if len(neg_issues):
    st.dataframe(neg_issues, use_container_width=True)
else:
    st.write("Not enough negative tweets to extract themes.")

# Positive hashtags
st.subheader("Positive-leaning hashtags")
if "text" in all_df.columns:
    pos_ht = (all_df[all_df["sentiment"]=="Positive"]["text"]
              .astype(str).str.findall(r"#\w+").explode().value_counts().head(15).reset_index())
    pos_ht.columns = ["hashtag", "count"]
    if len(pos_ht):
        fig_pos_ht = px.bar(pos_ht, x="hashtag", y="count", title="Top positive hashtags")
        st.plotly_chart(fig_pos_ht, use_container_width=True)
    else:
        st.write("No hashtags detected.")
else:
    st.write("Text column missing.")

st.divider()

# -------------------
# Recommendations (rule-based)
# -------------------
st.header("🧭 Recommended actions")
recs = []
total = len(all_df)
pos_rate = (all_df["sentiment"]=="Positive").mean()*100 if total else 0
neg_rate = (all_df["sentiment"]=="Negative").mean()*100 if total else 0

if total == 0:
    st.write("No data yet.")
else:
    if neg_rate >= 35:
        top_issues = ", ".join(neg_issues["bigram"].head(3).tolist())
        recs.append(f"High negative share ({neg_rate:.0f}%). Address: {top_issues}. Post clarification/update within 24–48h.")
    if pos_rate >= 60:
        recs.append("Strong positive share — amplify winning creatives, messaging tone, and timing.")
    if "brand" in all_df.columns and "SoP" in locals():
        best_brand = sop.iloc[0]["brand"] if not sop.empty else None
        if best_brand:
            recs.append(f"Benchmark best performer: **{best_brand}**. Mirror their tone/hashtags in the next sprint.")
    if 'pos_ht' in locals() and len(pos_ht):
        tags = ", ".join(pos_ht["hashtag"].head(5).tolist())
        recs.append(f"Use more of these positive hashtags: {tags}.")

    if recs:
        for r in recs: st.markdown(f"- {r}")
    else:
        st.write("No urgent actions. Maintain strategy and monitor weekly.")
 
