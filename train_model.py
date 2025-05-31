import nltk
nltk.download('punkt')
nltk.download('stopwords')

import pandas as pd
import string
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Preprocess function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    return " ".join(tokens)

# Load data
df = pd.read_csv(r'archive (3)\reviews_data.csv')
print("Columns in dataset:", df.columns)

# Clean text
df['cleaned'] = df['Review'].apply(preprocess_text)

# Create sentiment labels: 1 for rating >= 4, else 0
df['sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

# Features and target
X = df['cleaned']
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Handle imbalance
print("Class distribution in 'sentiment':")
print(y_train.value_counts())

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved successfully.")

# Function to predict new input
def predict_sentiment(text):
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return sentiment

# Example usage
if __name__ == "__main__":
    print("Starting test prediction...")
    test_review = "The coffee was amazing and the service was excellent!"
    print("Test review sentiment:", predict_sentiment(test_review))
