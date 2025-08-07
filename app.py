import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Twitter Brand Sentiment", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/Dataset-Train.csv")
    # Rename columns to simple names (change these if your CSV differs)
    df = df.rename(columns={
        "tweet_text": "text",
        "emotion_in_tweet_is_directed_at": "brand",
        "is_there_an_emotion_directed_at_a_brand_or_product": "sentiment"
    })
    df = df.dropna(subset=["text", "sentiment"])
    return df

df = load_data()

st.title("Twitter Brand Sentiment Dashboard")
st.caption("Visualize sentiment and keywords from brand-related tweets")

# ---- Sidebar filters ----
st.sidebar.header("Filters")
brands = ["All"] + sorted([b for b in df["brand"].dropna().unique()])
brand_sel = st.sidebar.selectbox("Brand", brands, index=0)
sentiments = df["sentiment"].dropna().unique().tolist()
sent_sel = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)

# Apply filters
f = df.copy()
if brand_sel != "All":
    f = f[f["brand"] == brand_sel]
if sent_sel:
    f = f[f["sentiment"].isin(sent_sel)]

# ---- KPI cards ----
col1, col2, col3 = st.columns(3)
col1.metric("Tweets", len(f))
col2.metric("Brands in view", f["brand"].nunique() if "brand" in f else 0)
col3.metric("Pos/Neg/Neu", "/".join([
    str((f["sentiment"]=="Positive").sum()),
    str((f["sentiment"]=="Negative").sum()),
    str((f["sentiment"]=="Neutral").sum())
]))

# ---- Sentiment distribution ----
st.subheader("Sentiment distribution")
sent_counts = f["sentiment"].value_counts().reset_index()
sent_counts.columns = ["sentiment", "count"]
fig = px.bar(sent_counts, x="sentiment", y="count", text="count", title="Sentiment counts")
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# ---- Brand comparison (if All brands) ----
if brand_sel == "All" and "brand" in f:
    st.subheader("Top brands by positive %")
    tmp = f.copy()
    tmp["is_pos"] = (tmp["sentiment"] == "Positive").astype(int)
    top = (tmp.groupby("brand")["is_pos"].mean()*100).sort_values(ascending=False).head(10).reset_index()
    fig2 = px.bar(top, x="brand", y="is_pos", title="Positive sentiment (%) by brand")
    st.plotly_chart(fig2, use_container_width=True)

# ---- Word clouds ----
st.subheader("Keyword clouds by sentiment")
for s in ["Positive", "Negative"]:
    text_blob = " ".join(f[f["sentiment"] == s]["text"].astype(str).tolist())
    if len(text_blob) > 0:
        wc = WordCloud(width=1000, height=400, background_color="white").generate(text_blob)
        st.markdown(f"**{s}**")
        fig_wc, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc, use_container_width=True)

# ---- Sample tweets ----
st.subheader("Sample tweets")
st.dataframe(f[["brand", "sentiment", "text"]].head(30))
