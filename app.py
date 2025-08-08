# app.py — Social Media Brand Sentiment (Beginner-friendly, Actionable)
import os, re
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Social Media Sentiment BI", layout="wide")
st.title("📊 Social Media Brand Sentiment — BI Dashboard")
st.info("Build v10 — tailored for 'Dataset-Train.csv' + 'Dataset - Test.csv'")

# --- VADER sentiment (pre-trained) ---
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
    out = []
    for t in texts:
        c = sia.polarity_scores(str(t))["compound"]
        out.append("Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral"))
    return out

# --- load CSVs (supports your two filenames) ---
os.makedirs("data", exist_ok=True)

def read_csv_safe(path_label_pairs):
    for path, label in path_label_pairs:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
                st.success(f"Loaded {label}: {path}")
                return df
            except Exception as e:
                st.warning(f"Could not read {path}: {e}")
    return None

train_df = read_csv_safe([
    ("data/Dataset-Train.csv", "TRAIN"),
    ("data/train.csv", "TRAIN"),
    ("data/Train.csv", "TRAIN"),
])

test_df = read_csv_safe([
    ("data/Dataset - Test.csv", "TEST"),
    ("data/Dataset-Test.csv", "TEST"),
    ("data/test.csv", "TEST"),
    ("data/Test.csv", "TEST"),
])

with st.expander("📁 What the app sees"):
    st.caption(f"Files in /data: {os.listdir('data') if os.path.exists('data') else 'no data folder'}")

if train_df is None and test_df is None:
    st.error("No CSVs found. Put files in /data or upload below.")
    up_train = st.file_uploader("Upload TRAIN CSV", type=["csv"], key="u_train")
    up_test  = st.file_uploader("Upload TEST CSV", type=["csv"], key="u_test")
    if up_train:
        with open("data/train.csv","wb") as f: f.write(up_train.read())
        st.success("Saved train.csv — click Rerun.")
    if up_test:
        with open("data/test.csv","wb") as f: f.write(up_test.read())
        st.success("Saved test.csv — click Rerun.")
    st.stop()

# --- basic column rename to have `text` and optional `brand` ---
def normalize_cols(df):
    if df is None: return None
    mapping = {"tweet_text":"text","Tweet":"text","Text":"text","text_clean":"text"}
    df = df.rename(columns={k:v for k,v in mapping.items() if k in df.columns})
    if "emotion_in_tweet_is_directed_at" in df.columns and "brand" not in df.columns:
        df = df.rename(columns={"emotion_in_tweet_is_directed_at":"brand"})
    return df

train_df = normalize_cols(train_df)
test_df  = normalize_cols(test_df)

# --- quick check for date columns ---
def add_date_cols(df):
    if df is None: return None
    for cand in ["created_at","tweet_created_at","date","timestamp","time"]:
        if cand in df.columns:
            df = df.rename(columns={cand:"created_at"})
            break
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        if df["created_at"].notna().any():
            df["day"] = df["created_at"].dt.date
    return df

train_df = add_date_cols(train_df)
test_df  = add_date_cols(test_df)

# --- fill brand if exists ---
if train_df is not None and "brand" in train_df.columns:
    train_df["brand"] = train_df["brand"].fillna("Unknown")
if test_df is not None and "brand" in test_df.columns:
    test_df["brand"] = test_df["brand"].fillna("Unknown")

# --- ensure we have text somewhere ---
if train_df is not None and "text" not in train_df.columns and "text" in (test_df.columns if test_df is not None else []):
    st.warning("TRAIN has no text column; proceeding with TEST only.")
    train_df = None

if (train_df is None or "text" not in train_df.columns) and (test_df is None or "text" not in test_df.columns):
    st.error("Neither file has a 'text' column (like Tweet/Text). Please adjust column names.")
    st.stop()

# --- quick model evaluation if TRAIN has labels ---
st.header("✅ Quick Model Check (train split)")
if train_df is not None and "sentiment" in train_df.columns and train_df["sentiment"].notna().any():
    labeled = train_df.dropna(subset=["text","sentiment"])
    if len(labeled) > 20:
        tr, val = train_test_split(labeled, test_size=0.2, random_state=42)
        y_true = val["sentiment"].astype(str)
        y_pred = predict_sentiment(val["text"].tolist())
        st.text("Classification report:")
        st.text(classification_report(y_true, y_pred))
    else:
        st.info("Not enough labeled rows to evaluate.")
else:
    st.info("Train has no usable labels → skipping evaluation.")

# --- predict sentiments where missing ---
def ensure_sentiment(df):
    if df is None: return None
    if "sentiment" not in df.columns or df["sentiment"].isna().all():
        df["sentiment"] = predict_sentiment(df["text"].tolist())
    return df

# If train_df exists but empty, drop it
if train_df is not None:
    if train_df.shape[1]==0 or len(train_df)==0:
        train_df = None

train_df = ensure_sentiment(train_df) if train_df is not None and "text" in train_df.columns else None
test_df  = ensure_sentiment(test_df)  if test_df  is not None and "text"  in test_df.columns  else None

# --- merge for BI ---
frames = []
if train_df is not None:
    train_df["source"] = "Train"
    frames.append(train_df)
if test_df is not None:
    test_df["source"] = "Test"
    frames.append(test_df)

all_df = pd.concat(frames, ignore_index=True)

st.divider()
st.header("📊 Core Insights")

# Overall sentiment pie
sent_counts = all_df["sentiment"].astype(str).value_counts().reset_index()
sent_counts.columns = ["Sentiment","Count"]
fig_pie = px.pie(sent_counts, names="Sentiment", values="Count", title="Overall Sentiment")
st.plotly_chart(fig_pie, use_container_width=True)

# Top-5 brands stacked bar + SoP if brand present
if "brand" in all_df.columns:
    st.subheader("Top 5 brands — sentiment split")
    top5 = all_df["brand"].value_counts().head(5).index
    small = all_df[all_df["brand"].isin(top5)]
    brand_sent = small.groupby(["brand","sentiment"]).size().reset_index(name="count")
    fig_bar = px.bar(brand_sent, x="brand", y="count", color="sentiment", barmode="stack",
                     title="Sentiment by brand (Top 5)")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Share of Positive (%) by brand")
    g = all_df.groupby(["brand","sentiment"]).size().reset_index(name="n")
    pivot = g.pivot(index="brand", columns="sentiment", values="n").fillna(0)
    pivot["SoP"] = (pivot.get("Positive",0) / pivot.sum(axis=1) * 100).round(1)
    sop = pivot["SoP"].sort_values(ascending=False).head(10).reset_index()
    fig_sop = px.bar(sop, x="brand", y="SoP", title="Top brands by Share of Positive (SoP)")
    st.plotly_chart(fig_sop, use_container_width=True)
else:
    st.info("No 'brand' column found — brand comparisons skipped.")

# Trend over time only if a date exists
if "day" in all_df.columns and all_df["day"].notna().any():
    st.subheader("Sentiment trend over time")
    trend = all_df.dropna(subset=["day"]).groupby(["day","sentiment"]).size().reset_index(name="count")
    fig_trend = px.line(trend, x="day", y="count", color="sentiment", markers=True, title="Daily sentiment counts")
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.caption("No timestamp found → add a date column to unlock trend views.")

st.divider()
st.header("🔎 Drivers & Guidance")

# Negative themes via bigrams (what to fix)
def top_bigrams(texts, n=12):
    pairs = []
    for t in texts:
        words = re.findall(r"[a-z]{3,}", str(t).lower())
        pairs += list(zip(words, words[1:]))
    counts = Counter(pairs).most_common(n)
    return pd.DataFrame([(" ".join(k), v) for k,v in counts], columns=["bigram","count"])

neg_issues = pd.DataFrame(columns=["bigram","count"])
if "text" in all_df.columns:
    neg_issues = top_bigrams(all_df[all_df["sentiment"]=="Negative"]["text"])
st.subheader("Top negative themes (bigrams)")
if len(neg_issues):
    st.dataframe(neg_issues, use_container_width=True)
else:
    st.write("Not enough negative tweets to extract themes.")

# Positive hashtags (what to amplify)
st.subheader("Positive-leaning hashtags")
if "text" in all_df.columns:
    pos_ht = (all_df[all_df["sentiment"]=="Positive"]["text"]
              .astype(str).str.findall(r"#\\w+").explode().value_counts().head(15).reset_index())
    pos_ht.columns = ["hashtag","count"]
    if len(pos_ht):
        fig_pos_ht = px.bar(pos_ht, x="hashtag", y="count", title="Top positive hashtags")
        st.plotly_chart(fig_pos_ht, use_container_width=True)
    else:
        st.write("No hashtags detected.")
else:
    st.write("Text column missing.")

st.divider()
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
        recs.append(f"High negative share ({neg_rate:.0f}%). Address: {top_issues}. Publish clarification within 24–48h.")
    if pos_rate >= 60:
        recs.append("Strong positive share — amplify winning creatives, tone, and timing.")
    if "brand" in all_df.columns:
        if 'sop' in locals() and not sop.empty and sop['SoP'].iloc[0] >= 60:
            recs.append(f"Benchmark best performer: {sop.iloc[0]['brand']}. Mirror their tone/hashtags in the next sprint.")
    if 'pos_ht' in locals() and len(pos_ht):
        tags = ", ".join(pos_ht['hashtag'].head(5).tolist())
        recs.append(f"Use more of these positive hashtags: {tags}.")

    if recs:
        for r in recs:
            st.markdown(f"- {r}")
    else:
        st.write("No urgent actions detected. Maintain strategy and monitor weekly.")
