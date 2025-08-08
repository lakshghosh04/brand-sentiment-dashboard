# app.py — Social Media Brand Sentiment Dashboard (Beginner-friendly)
import os, glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="📊 Social Media Brand Sentiment", layout="wide")
st.title("📊 Social Media Brand Sentiment Dashboard")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_data_folder():
    os.makedirs("data", exist_ok=True)

def find_csv(preferred_list):
    """Try preferred names first, else pick any CSV under /data."""
    for path in preferred_list:
        if os.path.exists(path):
            return path
    any_csvs = glob.glob("data/**/*.csv", recursive=True)
    return any_csvs[0] if any_csvs else None

def normalize_columns(df):
    """Rename common column names to: text, brand, sentiment."""
    rename_map = {
        "tweet_text": "text", "Tweet": "text", "Text": "text", "text_clean": "text",
        "emotion_in_tweet_is_directed_at": "brand", "brand": "brand", "Brand": "brand",
        "product": "brand", "Product": "brand",
        "is_there_an_emotion_directed_at_a_brand_or_product": "sentiment",
        "Sentiment": "sentiment", "label": "sentiment", "polarity": "sentiment"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

def normalize_sentiment_text(s: pd.Series) -> pd.Series:
    """Map various labels to Positive/Negative/Neutral."""
    def map_one(v):
        if pd.isna(v): return np.nan
        x = str(v).strip().lower()
        if "pos" in x: return "Positive"
        if "neg" in x: return "Negative"
        if "neu" in x or x in {"neutral", "none", ""}: return "Neutral"
        return "Neutral"
    return s.apply(map_one)

# -----------------------------------------------------------------------------
# Pre-trained sentiment (VADER)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Load train/test
# -----------------------------------------------------------------------------
ensure_data_folder()

train_path = find_csv([
    "data/train.csv", "data/Train.csv", "data/Dataset-Train.csv", "data/Dataset - Train.csv"
])
test_path = find_csv([
    "data/test.csv", "data/Test.csv", "data/Dataset-Test.csv", "data/Dataset - Test.csv"
])

with st.expander("📁 Data status"):
    st.write(f"Found train: **{train_path or 'None'}**")
    st.write(f"Found test: **{test_path or 'None'}**")
    st.write("Files in /data:", os.listdir("data") if os.path.exists("data") else "no data folder")

if not train_path:
    st.error("No train CSV found in /data. Upload below or add it to repo.")
    up = st.file_uploader("Upload TRAIN CSV", type=["csv"])
    if up:
        with open("data/train.csv", "wb") as f:
            f.write(up.read())
        st.success("Uploaded train.csv → Click 'Rerun' at the top.")
    st.stop()

try:
    train_df = pd.read_csv(train_path)
except Exception as e:
    st.error(f"Could not read {train_path}: {e}")
    st.stop()

test_df = None
if test_path:
    try:
        test_df = pd.read_csv(test_path)
    except Exception as e:
        st.warning(f"Could not read test file ({test_path}). Continuing without it.")
        test_df = None

train_df = normalize_columns(train_df)
if test_df is not None:
    test_df = normalize_columns(test_df)

if "text" not in train_df.columns:
    st.error("Train file must have a 'text' column. Rename in CSV or update mapping.")
    st.stop()

if "brand" in train_df.columns:
    train_df["brand"] = train_df["brand"].fillna("Unknown")
if test_df is not None and "brand" in test_df.columns:
    test_df["brand"] = test_df["brand"].fillna("Unknown")

if "sentiment" in train_df.columns:
    train_df["sentiment"] = normalize_sentiment_text(train_df["sentiment"])
if test_df is not None and "sentiment" in test_df.columns:
    test_df["sentiment"] = normalize_sentiment_text(test_df["sentiment"])

# -----------------------------------------------------------------------------
# Model Evaluation
# -----------------------------------------------------------------------------
st.header("✅ Model Evaluation")
if "sentiment" in train_df.columns and train_df["sentiment"].notna().any():
    labeled = train_df.dropna(subset=["text", "sentiment"]).copy()
    tr, val = train_test_split(labeled, test_size=0.2, random_state=42)

    y_true = val["sentiment"].astype(str)
    y_pred = predict_sentiment(val["text"].tolist())

    st.text("Classification Report")
    st.text(classification_report(y_true, y_pred))

    labels = sorted(set(y_true.unique()) | set(pd.Series(y_pred).unique()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    st.write("Confusion Matrix")
    st.dataframe(pd.DataFrame(cm, index=[f"Actual_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels]))
else:
    st.info("No sentiment labels in train → skipping evaluation.")

# -----------------------------------------------------------------------------
# Predict on unlabeled test
# -----------------------------------------------------------------------------
if test_df is not None:
    if "sentiment" not in test_df.columns or test_df["sentiment"].isna().all():
        test_df["sentiment"] = predict_sentiment(test_df["text"].tolist())

train_df["source"] = "Train"
if test_df is not None:
    test_df["source"] = "Test"
    all_df = pd.concat([train_df, test_df], ignore_index=True)
else:
    all_df = train_df.copy()

# -----------------------------------------------------------------------------
# Dashboard Visualizations
# -----------------------------------------------------------------------------
st.header("📊 Sentiment Analysis Insights")

# Overall sentiment pie chart
sent_counts = all_df["sentiment"].astype(str).value_counts().reset_index()
sent_counts.columns = ["Sentiment", "Count"]
fig_pie = px.pie(sent_counts, names="Sentiment", values="Count", title="Overall Sentiment Distribution", color="Sentiment")
st.plotly_chart(fig_pie, use_container_width=True)

# Top brands sentiment split (if brand present)
if "brand" in all_df.columns:
    st.subheader("Top 5 Brands Sentiment Split")
    top_brands = all_df["brand"].value_counts().head(5).index
    brand_sentiment = all_df[all_df["brand"].isin(top_brands)].groupby(["brand", "sentiment"]).size().reset_index(name="count")
    fig_brand = px.bar(brand_sentiment, x="brand", y="count", color="sentiment", barmode="stack", title="Sentiment Split for Top 5 Brands")
    st.plotly_chart(fig_brand, use_container_width=True)

# -----------------------------------------------------------------------------
# Strategy Suggestions
# -----------------------------------------------------------------------------
st.header("💡 Strategy Suggestions")
total = len(all_df)
pos_rate = (all_df["sentiment"].astype(str) == "Positive").mean() * 100
neg_rate = (all_df["sentiment"].astype(str) == "Negative").mean() * 100

if total == 0:
    st.write("No data to analyze yet.")
else:
    if neg_rate >= 35:
        st.write("⚠️ High negative sentiment → focus on addressing complaints & service issues.")
    if pos_rate >= 60:
        st.write("✅ Strong positive sentiment → amplify successful campaigns & repeat high-performing strategies.")
    st.write("📌 Consider adding a timestamp column for trend tracking over time.")
