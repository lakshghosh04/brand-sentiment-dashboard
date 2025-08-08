# app.py — simple, robust, beginner-friendly
import os, glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="📊 Social Media Brand Sentiment", layout="wide")
st.title("📊 Social Media Brand Sentiment Dashboard")

# -----------------------------------------------------------------------------
# 0) Helpers
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
    """Rename common column names to simple ones we use: text, brand, sentiment."""
    rename_map = {
        "tweet_text": "text",
        "Tweet": "text",
        "Text": "text",
        "text_clean": "text",

        "emotion_in_tweet_is_directed_at": "brand",
        "brand": "brand",
        "Brand": "brand",
        "product": "brand",
        "Product": "brand",

        "is_there_an_emotion_directed_at_a_brand_or_product": "sentiment",
        "Sentiment": "sentiment",
        "label": "sentiment",
        "polarity": "sentiment",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

def normalize_sentiment_text(s: pd.Series) -> pd.Series:
    """Map weird label text to Positive/Negative/Neutral."""
    def map_one(v):
        if pd.isna(v): return np.nan
        x = str(v).strip().lower()
        if "pos" in x or x == "positive": return "Positive"
        if "neg" in x or x == "negative": return "Negative"
        if "neu" in x or x == "neutral":  return "Neutral"
        if "irrelevant" in x or "no emotion" in x or x in {"none", ""}: return "Neutral"
        return x.title() if x in {"positive","negative","neutral"} else "Neutral"
    return s.apply(map_one)

# -----------------------------------------------------------------------------
# 1) Pre-trained sentiment (VADER)
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
# 2) Load train/test with robust fallback + upload option
# -----------------------------------------------------------------------------
ensure_data_folder()

# Try to auto-find train/test
train_path = find_csv([
    "data/train.csv", "data/Train.csv",
    "data/Dataset-Train.csv", "data/Dataset - Train.csv"
])
test_path = find_csv([
    "data/test.csv", "data/Test.csv",
    "data/Dataset-Test.csv", "data/Dataset - Test.csv"
])

with st.expander("📁 Data status (click to expand)"):
    st.write(f"Found train: **{train_path or 'None'}**")
    st.write(f"Found test: **{test_path or 'None'}**")
    st.write("Files in /data:", os.listdir("data") if os.path.exists("data") else "no data folder")

# If no train found, let user upload now (so app never blocks you)
if not train_path:
    st.error("No train CSV found in /data. Upload it below or add it to the repo.")
    up = st.file_uploader("Upload TRAIN CSV", type=["csv"])
    if up:
        with open("data/train.csv", "wb") as f:
            f.write(up.read())
        st.success("Uploaded train.csv → Click 'Rerun' (⟳) at the top.")
    st.stop()

# Read train safely
try:
    train_df = pd.read_csv(train_path)
except Exception as e:
    st.error(f"Could not read {train_path}: {e}")
    st.stop()

# Read test (optional)
test_df = None
if test_path:
    try:
        test_df = pd.read_csv(test_path)
    except Exception as e:
        st.warning(f"Could not read test file ({test_path}). Continuing without it. Error: {e}")
        test_df = None

# Normalize columns
train_df = normalize_columns(train_df)
if test_df is not None:
    test_df = normalize_columns(test_df)

# Basic checks
if "text" not in train_df.columns:
    st.error("Train file must have a text column (e.g., 'tweet_text' → 'text'). Rename column in CSV or adjust code mapping.")
    st.stop()

# Optional brand fill
if "brand" in train_df.columns:
    train_df["brand"] = train_df["brand"].fillna("Unknown")
if test_df is not None and "brand" in test_df.columns:
    test_df["brand"] = test_df["brand"].fillna("Unknown")

# If labels exist, clean them
if "sentiment" in train_df.columns:
    train_df["sentiment"] = normalize_sentiment_text(train_df["sentiment"])
if test_df is not None and "sentiment" in test_df.columns:
    test_df["sentiment"] = normalize_sentiment_text(test_df["sentiment"])

# -----------------------------------------------------------------------------
# 3) Model Evaluation (simple, only if train has labels)
# -----------------------------------------------------------------------------
st.header("✅ Model Evaluation (on train split)")
if "sentiment" in train_df.columns and train_df["sentiment"].notna().any():
    # Keep only rows with both text + label
    labeled = train_df.dropna(subset=["text", "sentiment"]).copy()

    # Split (no stratify to keep it simple)
    tr, val = train_test_split(labeled, test_size=0.2, random_state=42)

    y_true = val["sentiment"].astype(str)
    y_pred = predict_sentiment(val["text"].tolist())

    st.text("Classification Report")
    st.text(classification_report(y_true, y_pred))

    # Confusion matrix with labels actually present
    labels = sorted(set(y_true.unique()) | set(pd.Series(y_pred).unique()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    st.write("Confusion Matrix")
    st.dataframe(pd.DataFrame(cm,
                              index=[f"Actual_{l}" for l in labels],
                              columns=[f"Pred_{l}" for l in labels]))
else:
    st.info("No 'sentiment' labels in train file → skipping evaluation step.")

# -----------------------------------------------------------------------------
# 4) Predict on unlabeled test (if provided)
# -----------------------------------------------------------------------------
if test_df is not None:
    if "sentiment" not in test_df.columns or test_df["sentiment"].isna().all():
        test_df["sentiment"] = predict_sentiment(test_df["text"].tolist())

# Merge for dashboard
train_df["source"] = "Train"
if test_df is not None:
    test_df["source"] = "Test"
    all_df = pd.concat([train_df, test_df], ignore_index=True)
else:
    all_df = train_df.copy()

# -----------------------------------------------------------------------------
# 5) Dashboard
# -----------------------------------------------------------------------------
st.header("📈 Sentiment Distribution")
sent_counts = all_df["sentiment"].astype(str).value_counts(dropna=False).reset_index()
sent_counts.columns = ["Sentiment", "Count"]
fig = px.bar(sent_counts, x="Sentiment", y="Count", text="Count", color="Sentiment", title="Sentiment counts")
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# Top brands (if present)
if "brand" in all_df.columns:
    st.header("🏆 Top Brands by Positive %")
    all_df["is_positive"] = (all_df["sentiment"].astype(str) == "Positive").astype(int)
    top = (all_df.groupby("brand")["is_positive"].mean().sort_values(ascending=False) * 100).head(10)
    st.dataframe(top.rename("Positive %").round(1))

# Word clouds
st.header("☁️ Word Clouds")
for s in ["Positive", "Negative"]:
    st.subheader(f"{s} Tweets")
    blob = " ".join(all_df[all_df["sentiment"].astype(str) == s]["text"].astype(str).tolist())
    if blob.strip():
        wc = WordCloud(width=900, height=400, background_color="white").generate(blob)
        fig_wc, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc, use_container_width=True)
    else:
        st.write("No tweets for this class.")

# Simple strategy tips
st.header("💡 Strategy Suggestions")
total = len(all_df)
pos_rate = (all_df["sentiment"].astype(str) == "Positive").mean() * 100
neg_rate = (all_df["sentiment"].astype(str) == "Negative").mean() * 100

if total == 0:
    st.write("No data to analyze yet.")
else:
    if neg_rate >= 35:
        st.write("⚠️ High negative sentiment → review negative word cloud; address top 3 issues in comms.")
    if pos_rate >= 60:
        st.write("✅ Strong positive sentiment → amplify winning themes and hashtags.")
    st.write("Tip: Add a date/timestamp column to analyze sentiment trends over time.")
