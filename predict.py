import joblib
import re
import string
import nltk

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
# ----------------------------
# Clean the input text
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]  # Remove stopwords
    text = ' '.join(words)
    print("[DEBUG] Cleaned text:", text)
    return text

# ----------------------------
# Load model and vectorizer
# ----------------------------
def load_model_and_vectorizer(model_path='models\sentiment_classifier_movie.pkl', vectorizer_path="vectorizer.pkl"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# ----------------------------
# Predict sentiment of a single review
# ----------------------------
def predict_sentiment(review, model, vectorizer):
    # Clean the review
    cleaned_review = clean_text(review)

    # Vectorize
    review_vector = vectorizer.transform([cleaned_review]).toarray()

    # Predict
    prediction = model.predict(review_vector)[0]  # Get the first (only) result

    # Map label to sentiment
    label_mapping = {1: 'Positive', 0: 'Negative'}
    return label_mapping[prediction]

# ----------------------------
# Main function for command line usage
# ----------------------------
if __name__ == "__main__":
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()

    # Input review from user
    review = input("Enter a movie review: ")

    # Predict sentiment
    sentiment = predict_sentiment(review, model, vectorizer)

    # Output result
    print(f"Predicted Sentiment: {sentiment}")
