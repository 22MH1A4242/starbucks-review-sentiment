import streamlit as st
import pandas as pd
import joblib
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    return " ".join(tokens)

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]
    return "🌟 Positive" if pred == 1 else "⚠️ Negative"

st.set_page_config(page_title="Starbucks Sentiment App", layout="centered")
st.title("☕ Starbucks Review Sentiment Analyzer")

mode = st.sidebar.selectbox("Choose mode", ["Single Review", "Batch Upload"])

if mode == "Single Review":
    st.write("Enter a Starbucks review below to see whether it's Positive or Negative.")
    user_input = st.text_area("📝 Your Review:")

    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a review.")
        else:
            sentiment = predict_sentiment(user_input)
            st.success(f"Predicted Sentiment: **{sentiment}**")

elif mode == "Batch Upload":
    st.write("Upload a CSV file with a column named 'Review' to predict sentiment for multiple reviews.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Review' not in df.columns:
            st.error("CSV file must contain a 'Review' column.")
        else:
            st.write(f"Loaded {len(df)} reviews.")
            df['Predicted Sentiment'] = df['Review'].apply(predict_sentiment)
            st.dataframe(df[['Review', 'Predicted Sentiment']])
