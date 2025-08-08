# app.py
import os, glob, re
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -------- Toggle for model choice --------
# Fast, lightweight default:
USE_HF = False  # set True to use HuggingFace twitter-roberta (needs transformers+torch)

st.set_page_config(page_title="Social Media Brand Sentiment", layout="wide")

# =========================
# Utilities
# =========================
def find_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    any_csvs = glob.glob("data/**/*.csv", recursive=True)
    return any_csvs[0] if any_csvs else None

def normalize_columns(df):
    # Try common column names and normalize to: text, brand, sentiment, created_at(optional)
    rename_map = {
        "tweet_text": "text",
        "Tweet": "text",
        "Text": "text",

        "emotion_in_tweet_is_directed_at": "brand",
        "brand": "brand",
        "Brand": "brand",
        "product": "brand",
        "Product": "brand",

        "is_there_an_emotion_directed_at_a_brand_or_product": "sentiment",
        "Sentiment": "sentiment",
        "label": "sentiment",
        "polarity": "sentiment",

        "tweet_created_at": "created_at",
        "created_at": "created_at",
        "date": "created_at",
        "timestamp": "created_at",
        "time": "created_at",
    }
    # Only rename keys that exist
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Minimal required: text
    if "text" not in df.columns:
        # Try a couple more guesses
        for c in ["content", "message"]:
            if c in df.columns:
                df = df.rename(columns={c: "text"})
                break
    return df

def parse_dates(df):
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        if df["created_at"].notna().any():
            df["date_only"] = df["created_at"].dt.date
        else:
            df.drop(columns=["created_at"], inplace=True, errors="ignore")
    return df

def top_terms(series: pd.Series, n=20):
    text = " ".join(series.astype(str)).lower()
    # keep hashtags and words (min 4 chars to cut noise)
    tokens = re.findall(r"#\w+|\b[a-z]{4,}\b", text)
    stop = {
        "https","http","rt","with","from","this","that","have","about","your","they","them","were",
        "been","their","there","will","could","would","should","what","when","where","which","into",
        "because","while","after","before","just","like","https","co"
    }
    tokens = [t for t in tokens if t not in stop]
    return pd.DataFrame(Counter(tokens).most_common(n), columns=["term","freq"])

# =========================
# Data Loading
# =========================
@st.cache_data
def load_train_test():
    # Try common names; fall back to any csv under data/
    train_path = find_file([
        "data/train.csv", "data/Train.csv", "data/Dataset-Train.csv", "data/Dataset - Train.csv"
    ])
    test_path = find_file([
        "data/test.csv", "data/Test.csv", "data/Dataset-Test.csv", "data/Dataset - Test.csv"
    ])

    train_df = pd.read_csv(train_path) if train_path else None
    test_df  = pd.read_csv(test_path)  if test_path  else None

    if train_df is None and test_df is None:
        st.error("Could not find train/test CSVs in the data/ folder.")
        st.stop()

    if train_df is not None:
        train_df = normalize_columns(train_df)
        train_df = parse_dates(train_df)

    if test_df is not None:
        test_df = normalize_columns(test_df)
        test_df = parse_dates(test_df)

    return train_df, test_df, train_path, test_path

train_df, test_df, train_path, test_path = load_train_test()
st.caption(f"Loaded: train → {os.path.basename(train_path) if train_path else 'None'} | test → {os.path.basename(test_path) if test_path else 'None'}")

# Ensure minimal columns
if train_df is not None and "text" not in train_df.columns:
    st.error("Train file must contain a text column (e.g., tweet_text → text).")
    st.stop()
if test_df is not None and "text" not in test_df.columns:
    st.warning("Test file has no 'text' column after normalization. Test data will be ignored.")
    test_df = None

# Fill brand if exists
if train_df is not None and "brand" in train_df.columns:
    train_df["brand"] = train_df["brand"].fillna("Unknown")
if test_df is not None and "brand" in test_df.columns:
    test_df["brand"] = test_df["brand"].fillna("Unknown")

# =========================
# Sentiment Models (pre-trained)
# =========================
@st.cache_resource
def load_vader():
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    from nltk.sentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_hf_pipeline():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def infer_sentiment_vader(texts):
    sia = load_vader()
    def to_label(comp):
        if comp >= 0.05: return "Positive"
        if comp <= -0.05: return "Negative"
        return "Neutral"
    scores = [sia.polarity_scores(str(t))["compound"] for t in texts]
    return [to_label(s) for s in scores]

def infer_sentiment_hf(texts):
    clf = load_hf_pipeline()
    preds = clf(list(map(str, texts)))
    # Model labels may be NEGATIVE/NEUTRAL/POSITIVE or LABEL_0/1/2
    label_map = {"NEGATIVE":"Negative","NEUTRAL":"Neutral","POSITIVE":"Positive",
                 "LABEL_0":"Negative","LABEL_1":"Neutral","LABEL_2":"Positive"}
    return [label_map.get(p["label"], p["label"]) for p in preds]

def infer_sentiment(texts):
    return infer_sentiment_hf(texts) if USE_HF else infer_sentiment_vader(texts)

# =========================
# Tabs: Dashboard | Model Validation
# =========================
tab_dash, tab_eval = st.tabs(["📊 Dashboard", "✅ Model Validation"])

# ===========================================================
# MODEL VALIDATION (on labeled data only)
# ===========================================================
with tab_eval:
    st.subheader("Model Validation on Labeled Data")
    if train_df is None or "sentiment" not in train_df.columns:
        st.info("No gold sentiment labels found in train data. Validation skipped.")
    else:
        # Clean labeled rows only
        labeled = train_df.dropna(subset=["text","sentiment"]).copy()

        # Stratified split
        train_split, val_split = train_test_split(
            labeled, test_size=0.2, random_state=42, stratify=labeled["sentiment"]
        )

        st.write(f"Train size: {len(train_split)} | Validation size: {len(val_split)}")

        with st.spinner("Running pre-trained model on validation split..."):
            val_preds = infer_sentiment(val_split["text"].tolist())

        # Align labels if needed (capitalize etc.)
        y_true = val_split["sentiment"].astype(str).str.title()
        y_pred = pd.Series(val_preds).astype(str).str.title()

        # Metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        st.write("**Classification report**")
        st.dataframe(pd.DataFrame(report).T.round(3))

        # Confusion matrix
        st.write("**Confusion matrix**")
        labels = ["Positive","Neutral","Negative"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True,
                           color_continuous_scale="Blues", aspect="auto",
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("""
        **Interpretation tips:**  
        - If Neutral is often misclassified as Positive/Negative, consider rules like confidence thresholds or human review.  
        - If Negative recall is low, scan top negative terms to tune thresholds or try the HF model (set `USE_HF=True`).  
        """)

# ===========================================================
# DASHBOARD (merge labeled + predicted)
# ===========================================================
with tab_dash:
    st.title("Social Media Brand Sentiment Dashboard")
    st.caption("Actionable insights from Twitter-like brand sentiment data (train/test merged).")

    # Prepare train view:
    # - If train has labels, keep them as-is in 'sentiment'
    # - If not, infer
    train_view = None
    if train_df is not None:
        train_view = train_df.copy()
        if "sentiment" not in train_view.columns or train_view["sentiment"].isna().all():
            with st.spinner("Inferring sentiment on train (no labels found)..."):
                train_view["sentiment"] = infer_sentiment(train_view["text"].tolist())
        train_view["source"] = "train"

    # Prepare test view:
    # - If test has labels, use them
    # - If unlabeled, infer
    test_view = None
    if test_df is not None:
        test_view = test_df.copy()
        if "sentiment" not in test_view.columns or test_view["sentiment"].isna().all():
            with st.spinner("Inferring sentiment on test (unlabeled)..."):
                test_view["sentiment"] = infer_sentiment(test_view["text"].tolist())
        test_view["source"] = "test"

    # Merge
    frames = [d for d in [train_view, test_view] if d is not None]
    if not frames:
        st.error("No data available to display.")
        st.stop()
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["text","sentiment"])
    if "brand" in all_df.columns:
        all_df["brand"] = all_df["brand"].fillna("Unknown")

    # Sidebar filters
    st.sidebar.header("Filters")
    sources = ["All"] + sorted(all_df["source"].unique().tolist())
    src_sel = st.sidebar.selectbox("Source", sources, index=0)

    brand_options = ["All"]
    if "brand" in all_df.columns:
        brand_options += sorted(all_df["brand"].dropna().unique().tolist())
    brand_sel = st.sidebar.selectbox("Brand", brand_options, index=0)

    sentiments = sorted(all_df["sentiment"].dropna().astype(str).str.title().unique().tolist())
    sent_sel = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)

    # Date filter if present
    date_col = "date_only" if "date_only" in all_df.columns else None
    date_range = None
    if date_col and all_df[date_col].notna().any():
        min_d, max_d = all_df[date_col].min(), all_df[date_col].max()
        date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    # Apply filters
    f = all_df.copy()
    if src_sel != "All":
        f = f[f["source"] == src_sel]
    if brand_sel != "All" and "brand" in f.columns:
        f = f[f["brand"] == brand_sel]
    if sent_sel:
        f = f[f["sentiment"].astype(str).str.title().isin(sent_sel)]
    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2 and date_col:
        start, end = date_range
        f = f[(f[date_col] >= start) & (f[date_col] <= end)]

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total tweets", len(f))
    col2.metric("Unique brands", f["brand"].nunique() if "brand" in f.columns else 0)
    pos = (f["sentiment"].astype(str).str.title() == "Positive").sum()
    neg = (f["sentiment"].astype(str).str.title() == "Negative").sum()
    neu = (f["sentiment"].astype(str).str.title() == "Neutral").sum()
    col3.metric("Pos / Neg / Neu", f"{pos} / {neg} / {neu}")
    if date_col:
        col4.metric("Date span", f"{f[date_col].min()} → {f[date_col].max()}")

    # Sentiment distribution
    st.subheader("Sentiment distribution")
    sent_counts = f["sentiment"].astype(str).str.title().value_counts().reset_index()
    sent_counts.columns = ["sentiment", "count"]
    fig = px.bar(sent_counts, x="sentiment", y="count", text="count", title="Sentiment counts")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # Brand comparison
    if "brand" in f.columns and f["brand"].notna().sum() > 0 and brand_sel == "All":
        st.subheader("Top brands by positive %")
        tmp = f.copy()
        tmp["is_pos"] = (tmp["sentiment"].astype(str).str.title() == "Positive").astype(int)
        brand_stats = (tmp.groupby("brand")["is_pos"].mean() * 100).sort_values(ascending=False)
        top = brand_stats.head(10).reset_index().rename(columns={"is_pos": "positive_pct"})
        if len(top) > 0:
            fig2 = px.bar(top, x="brand", y="positive_pct", title="Positive sentiment (%) by brand")
            fig2.update_layout(yaxis_title="Positive (%)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough brand data to compare yet.")

    # Sentiment trend (if dates)
    if date_col and f[date_col].notna().any():
        st.subheader("Sentiment over time")
        g = f.groupby([date_col, f["sentiment"].astype(str).str.title()]).size().reset_index(name="count")
        fig_time = px.line(g, x=date_col, y="count", color="sentiment", markers=True,
                           title="Sentiment trend by day")
        st.plotly_chart(fig_time, use_container_width=True)

    # Keyword clouds
    st.subheader("Keyword clouds by sentiment")
    cols_wc = st.columns(2)
    for i, s in enumerate(["Positive", "Negative"]):
        sub = f[f["sentiment"].astype(str).str.title() == s]
        with cols_wc[i]:
            st.markdown(f"**{s}**")
            if len(sub) > 0:
                text_blob = " ".join(sub["text"].astype(str).tolist())
                wc = WordCloud(width=1000, height=400, background_color="white").generate(text_blob)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc, use_container_width=True)
            else:
                st.write("No tweets in view.")

    # Top terms by sentiment
    st.subheader("Top terms by sentiment")
    cols_tt = st.columns(3)
    for i, s in enumerate(["Positive", "Negative", "Neutral"]):
        sub = f[f["sentiment"].astype(str).str.title() == s]
        with cols_tt[i]:
            st.markdown(f"**{s}**")
            if len(sub) > 0:
                st.dataframe(top_terms(sub["text"], n=15), use_container_width=True, height=350)
            else:
                st.write("No tweets in view.")

    # Hashtags
    st.subheader("Top hashtags")
    hashtags = f["text"].astype(str).str.findall(r"#\w+").explode()
    if hashtags.notna().sum() > 0:
        ht_counts = hashtags.value_counts().head(20).reset_index()
        ht_counts.columns = ["hashtag","count"]
        fig_ht = px.bar(ht_counts, x="hashtag", y="count", title="Top 20 hashtags")
        st.plotly_chart(fig_ht, use_container_width=True)
    else:
        st.write("No hashtags found in current selection.")

    # Strategy suggestions (simple rules)
    st.subheader("Strategy suggestions")
    suggestions = []
    total = len(f)
    if total > 0:
        pos_rate = pos / total * 100 if total else 0
        neg_rate = neg / total * 100 if total else 0
        if neg_rate >= 35:
            suggestions.append("High negative sentiment detected — review 'What people complain about' terms and address top 3 issues in comms.")
        if pos_rate >= 60:
            suggestions.append("Strong positive sentiment — amplify winning themes and hashtags in upcoming posts.")
        # Topic-like hints from top terms
        pos_terms = top_terms(f[f["sentiment"].astype(str).str.title()=="Positive"]["text"], n=5)
        neg_terms = top_terms(f[f["sentiment"].astype(str).str.title()=="Negative"]["text"], n=5)
        if len(pos_terms):
            suggestions.append(f"Leverage positive hooks: {', '.join(pos_terms['term'].tolist())}.")
        if len(neg_terms):
            suggestions.append(f"Mitigate pain points: {', '.join(neg_terms['term'].tolist())}.")
        if "brand" in f.columns and brand_sel == "All":
            # Highlight top brand by positive%
            tmp = f.copy()
            tmp["is_pos"] = (tmp["sentiment"].astype(str).str.title()=="Positive").astype(int)
            brand_stats = (tmp.groupby("brand")["is_pos"].mean()*100).sort_values(ascending=False)
            if len(brand_stats) >= 1:
                topb = brand_stats.index[0]
                suggestions.append(f"Benchmark: '{topb}' has the highest positive% — analyze their messaging/hashtags for cues.")
    if suggestions:
        for s in suggestions:
            st.markdown(f"- {s}")
    else:
        st.write("Not enough data for suggestions yet.")

    # Sample tweets
    st.subheader("Sample tweets")
    cols_show = [c for c in ["source","brand","sentiment","text","created_at"] if c in f.columns]
    st.dataframe(f[cols_show].head(40), use_container_width=True)
