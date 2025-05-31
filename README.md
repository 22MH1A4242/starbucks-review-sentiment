# ☕ Starbucks Review Sentiment Analysi

A machine learning-based sentiment analysis application that predicts whether a Starbucks review is **Positive** or **Negative**.  
Built with Python, scikit-learn, NLTK for NLP preprocessing, and Streamlit for an interactive web app.

---

## Features

- **Single Review Analysis:** Enter a Starbucks review and get instant sentiment prediction.
- **Batch Review Analysis:** Upload a CSV file containing multiple reviews to predict sentiments in bulk.
- **Preprocessing:** Text cleaning with tokenization, stopword removal, and punctuation filtering.
- **Model:** Logistic Regression classifier with TF-IDF vectorization.
- **Interactive Web UI:** Streamlit app for easy user interaction.
- **Model Persistence:** Model and vectorizer saved using joblib for fast loading and predicti

## 🗂 Project Structure
 
 sentiment-analysis/
├── data/
├── notebooks/
├── model/
├── app/
├── utils/
├── requirements.txt
├── README.md


---

## Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/starbucks-sentiment-analyzer.git
   cd starbucks-sentiment-analyzer

   Create and activate a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies:

pip install -r requirements.txt
Download NLTK data (if not downloaded automatically):

import nltk
nltk.download('punkt')
nltk.download('stopwords')
Usage
Train the Model
Run the training script to preprocess data, train the sentiment model, and save artifacts:

python train_model.py
Run the Streamlit App
Start the interactive web app:

streamlit run app.py
Enter a single Starbucks review for prediction.

Or upload a CSV file with a "Review" column to analyze multiple reviews at once.

Deployment
You can deploy this Streamlit app online using:

Streamlit Community Cloud

Render

Railway

Dataset
The dataset used contains Starbucks reviews with columns like Review, Rating, and metadata.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
NLTK

scikit-learn

Streamlit

Contact
Created by Anjali Devi Medapati









