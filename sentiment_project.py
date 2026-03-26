import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

# ---------------- CLEANING FUNCTION ----------------
def mycleaning(doc):
    return re.sub("[^a-zA-Z ]","",doc).lower()

# ---------------- LOAD MODEL ----------------
model = joblib.load("sentiment_model.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🍔 Food Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.image("food_sentiment.jpeg")

st.sidebar.markdown("## 🍽️ Smart Food Review Analyzer")

st.sidebar.markdown("""
🔍 **What this app does:**
- Analyzes customer food reviews  
- Detects Positive 😊 or Negative 😡 sentiment  
- Helps businesses improve quality  

💡 **Why it matters:**
- Understand customer satisfaction  
- Improve restaurant services  
- Make data-driven decisions  

⚙️ **Tech Used:**
- Machine Learning (NLP)
- Scikit-learn
- Streamlit Dashboard
""")

st.sidebar.markdown("## 📊 Features")
st.sidebar.markdown("""
✔ Real-time sentiment prediction  
✔ Bulk review analysis  
✔ Confidence score  
✔ Data visualization  
✔ Download results  
""")

st.sidebar.markdown("## 👨‍💻 About Me")
st.sidebar.write("Aspiring Data Scientist | ML Enthusiast")

st.sidebar.markdown("## 📧 Contact")
st.sidebar.write("reviewsample@gmail.com")


# ---------------- SINGLE INPUT ----------------
st.subheader("📝 Single Review Prediction")

sample = st.text_input("Enter your review here")

if st.button("Predict Sentiment"):
    pred = model.predict([sample])
    prob = model.predict_proba([sample])

    col1, col2 = st.columns(2)

    with col1:
        if pred[0] == 0:
            st.error("Negative Review 😡")
        else:
            st.success("Positive Review 😊")

    with col2:
        confidence = np.max(prob)
        st.metric("Confidence Score", f"{confidence:.2f}")

# ---------------- BULK INPUT ----------------
st.subheader("📂 Bulk Review Analysis")

file = st.file_uploader("Upload CSV file", type=["csv", "txt"])

if file:
    df = pd.read_csv(file, names=["Review"])
    st.dataframe(df, use_container_width=True)

    if st.button("Analyze Reviews"):
        corpus = df["Review"]

        pred = model.predict(corpus)
        prob = np.max(model.predict_proba(corpus), axis=1)

        df["Sentiment"] = pred
        df["Confidence"] = prob

        df["Sentiment"] = df["Sentiment"].map({0: "Negative", 1: "Positive"})

        st.success("Analysis Complete ✅")
        st.dataframe(df, use_container_width=True)

        # ---------------- METRICS ----------------
        pos_count = (df["Sentiment"] == "Positive").sum()
        neg_count = (df["Sentiment"] == "Negative").sum()

        col1, col2 = st.columns(2)

        col1.metric("😊 Positive Reviews", pos_count)
        col2.metric("😡 Negative Reviews", neg_count)

        # ---------------- CHART ----------------
        st.subheader("📊 Sentiment Distribution")
        chart_data = df["Sentiment"].value_counts()
        st.bar_chart(chart_data)

        # ---------------- DOWNLOAD ----------------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Results",
            csv,
            "sentiment_results.csv",
            "text/csv"
        )