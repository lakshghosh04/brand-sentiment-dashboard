import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="Social Media Sentiment", layout="wide")
st.title("📊 Social Media Brand Sentiment (Simple)")

# --- load VADER once ---
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

# --- get data (simple paths + uploader fallback) ---
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
        st.success(f"Saved to {tmp_path}. Click ▶ Rerun on the top bar.")
        st.stop()
    return None

train_df = read_csv_or_upload("Train", "data/train.csv")
test_df  = read_csv_or_upload("Test (optional)", "data/test.csv")

if train_df is None:
    st.stop()

# --- minimal column rename so we have 'text' ---
rename_map = {"tweet_text": "text", "Tweet": "text", "Text": "text"}
train_df = train_df.rename(columns={k:v for k,v in rename_map.items() if k in train_df.columns})
if test_df is not None:
    test_df = test_df.rename(columns={k:v for k,v in rename_map.items() if k in test_df.columns})

if "text" not in train_df.columns:
    st.error("Your train file must have a text column (e.g., 'tweet_text' or 'text').")
    st.stop()

# --- optional: normalize brand column name ---
if "emotion_in_tweet_is_directed_at" in train_df.columns and "brand" not in train_df.columns:
    train_df = train_df.rename(columns={"emotion_in_tweet_is_directed_at": "brand"})
if test_df is not None and "emotion_in_tweet_is_directed_at" in test_df.columns and "brand" not in test_df.columns:
    test_df = test_df.rename(columns={"emotion_in_tweet_is_directed_at": "brand"})

if "brand" in train_df.columns:
    train_df["brand"] = train_df["brand"].fillna("Unknown")
if test_df is not None and "brand" in test_df.columns:
    test_df["brand"] = test_df["brand"].fillna("Unknown")

st.divider()

# --- simple evaluation if train has labels ---
st.header("✅ Quick Model Check (uses train split)")
if "sentiment" in train_df.columns:
    labeled = train_df.dropna(subset=["text", "sentiment"])
    if len(labeled) > 10:
        tr, val = train_test_split(labeled, test_size=0.2, random_state=42)
        y_true = val["sentiment"].astype(str)
        y_pred = predict_sentiment(val["text"].tolist())
        st.text("Classification report:")
        st.text(classification_report(y_true, y_pred))
    else:
        st.info("Not enough labeled rows to evaluate. Skipping.")
else:
    st.info("No 'sentiment' column in train data. Skipping evaluation (we'll predict sentiments below).")

st.divider()

# --- if test has no labels, predict them; do same for train if unlabeled ---
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

st.header("📊 Dashboard")

# --- overall sentiment pie ---
sent_counts = all_df["sentiment"].astype(str).value_counts().reset_index()
sent_counts.columns = ["Sentiment", "Count"]
fig_pie = px.pie(sent_counts, names="Sentiment", values="Count", title="Overall Sentiment")
st.plotly_chart(fig_pie, use_container_width=True)

# --- top 5 brands stacked bar (if brand exists) ---
if "brand" in all_df.columns:
    st.subheader("Top 5 brands: sentiment split")
    top5 = all_df["brand"].value_counts().head(5).index
    small = all_df[all_df["brand"].isin(top5)]
    brand_sent = small.groupby(["brand","sentiment"]).size().reset_index(name="count")
    fig_bar = px.bar(brand_sent, x="brand", y="count", color="sentiment", barmode="stack",
                     title="Sentiment by brand (Top 5)")
    st.plotly_chart(fig_bar, use_container_width=True)

# --- quick tips ---
st.header("💡 Strategy Tips")
total = len(all_df)
pos_rate = (all_df["sentiment"].astype(str) == "Positive").mean()*100 if total else 0
neg_rate = (all_df["sentiment"].astype(str) == "Negative").mean()*100 if total else 0

if total == 0:
    st.write("No data yet.")
else:
    if neg_rate >= 35:
        st.write("⚠️ High negative share → check complaints and address them in comms/service.")
    if pos_rate >= 60:
        st.write("✅ Strong positive share → amplify successful themes and creatives.")
    st.write("Pro tip: add a date column later to see trends over time.")
