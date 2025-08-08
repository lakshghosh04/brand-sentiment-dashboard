import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Load VADER sentiment model
# -----------------------------
@st.cache_resource
def load_vader():
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    from nltk.sentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

# Simple VADER prediction
def predict_sentiment(texts):
    sia = load_vader()
    results = []
    for t in texts:
        score = sia.polarity_scores(str(t))["compound"]
        if score >= 0.05:
            results.append("Positive")
        elif score <= -0.05:
            results.append("Negative")
        else:
            results.append("Neutral")
    return results

# -----------------------------
# Load data
# -----------------------------
st.title("📊 Social Media Brand Sentiment Dashboard")

train_path = "data/train.csv"
test_path = "data/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Make sure we have a text column
train_df = train_df.rename(columns={"tweet_text": "text", "Tweet": "text"})
test_df = test_df.rename(columns={"tweet_text": "text", "Tweet": "text"})

# Fill missing brand values if exist
if "brand" in train_df.columns:
    train_df["brand"] = train_df["brand"].fillna("Unknown")
if "brand" in test_df.columns:
    test_df["brand"] = test_df["brand"].fillna("Unknown")

# -----------------------------
# Evaluate model (if labels exist)
# -----------------------------
st.header("✅ Model Evaluation")
if "sentiment" in train_df.columns:
    # Split data
    train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=42)
    y_true = val_split["sentiment"]
    y_pred = predict_sentiment(val_split["text"].tolist())

    st.text("Classification Report")
    st.text(classification_report(y_true, y_pred))

    st.text("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred, labels=["Positive", "Negative", "Neutral"])
    st.write(pd.DataFrame(cm, index=["Pos", "Neg", "Neu"], columns=["Pos", "Neg", "Neu"]))
else:
    st.info("No sentiment labels in training data to evaluate.")

# -----------------------------
# Predict sentiment for unlabeled test data
# -----------------------------
if "sentiment" not in test_df.columns:
    test_df["sentiment"] = predict_sentiment(test_df["text"].tolist())

# Merge both datasets
train_df["source"] = "Train"
test_df["source"] = "Test"
all_data = pd.concat([train_df, test_df], ignore_index=True)

# -----------------------------
# Dashboard Insights
# -----------------------------
st.header("📈 Sentiment Distribution")
sent_counts = all_data["sentiment"].value_counts().reset_index()
sent_counts.columns = ["Sentiment", "Count"]
fig = px.bar(sent_counts, x="Sentiment", y="Count", text="Count", color="Sentiment")
st.plotly_chart(fig, use_container_width=True)

# Top brands by positive %
if "brand" in all_data.columns:
    st.header("🏆 Top Brands by Positive %")
    all_data["is_positive"] = (all_data["sentiment"] == "Positive").astype(int)
    brand_stats = all_data.groupby("brand")["is_positive"].mean().sort_values(ascending=False).head(10) * 100
    st.write(brand_stats)

# -----------------------------
# Word Clouds
# -----------------------------
st.header("☁️ Word Clouds")
for sentiment in ["Positive", "Negative"]:
    st.subheader(f"{sentiment} Tweets")
    text = " ".join(all_data[all_data["sentiment"] == sentiment]["text"].astype(str))
    if text.strip():
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.write("No tweets available.")

# -----------------------------
# Simple Strategy Suggestions
# -----------------------------
st.header("💡 Strategy Suggestions")
total = len(all_data)
pos_rate = (all_data["sentiment"] == "Positive").mean() * 100
neg_rate = (all_data["sentiment"] == "Negative").mean() * 100

if neg_rate > 30:
    st.write("⚠️ High negative sentiment detected. Investigate key complaints.")
if pos_rate > 60:
    st.write("✅ Strong positive sentiment. Promote themes that resonate well.")

st.success("Dashboard ready — filter, explore, and take action!")
