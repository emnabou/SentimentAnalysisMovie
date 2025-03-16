#src/data_preprocessing.py
#Loading dataset
#cleaning the text
#Tokenization ans stopword removal
#Vectorization
#Saving the processed data and vectorizer

import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import os

#Download the stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))


#--------------------- 1.Text Cleaning ---------------------
def clean_text(text):
    """
    Function to clean the text
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]  # Remove stopwords
    text = ' '.join(words)
    return text

#--------------------- 2. Preprocessing Function ---------------------
def preprocess_data(file_path):
    """
    Function to load the dataset and preprocess it
    """
    # Load the dataset
    print("Loading the dataset....")
    df = pd.read_csv(file_path, quotechar='"')
    print("[DEBUG] First few rows of raw data:\n", df.head())
    print("Dataset loaded successfully.")
    print("Cleaning text data....")
    df['clean_text'] = df['text'].apply(clean_text)
    print("[DEBUG] First few rows after cleaning:\n", df[['text', 'clean_text']].head())

    return df

#--------------------- 3. Vectorization Function ---------------------
def vectorize_text(text_series):
    """
    Vectorize text data using TF-IDF Vectorizer
    """
    print("Vectorizing the text data....")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_series)

    return X, vectorizer

#--------------------- 4. Main Run Function ---------------------
def run():
    """
    Full preprocessing pipeline: load, clean, vectorize and save
    """
    data_file="data\movie.csv"
    #Preprocess the data
    df=preprocess_data(data_file)
    #Vectorize the text data
    X, vectorizer=vectorize_text(df['clean_text'])
    print("[DEBUG] Vectorized feature matrix shape:", X.shape)
    #Encode labels

    print("Encoding target labels....")
    y = df['label'].values  # Target labels
    label_mapping = {1: 'Positive', 0: 'Negative'}
    print(y[:10])
    print("[DEBUG] Labels after encoding:", y.tolist())
    #save processed data
    os.makedirs("processed_Movie",exist_ok=True)
    joblib.dump((X,y), "processed_Movie/preprocessed_data.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Data preprocessing and vectorization completed.")
    return 0


#--------------------- 5. Run the script ---------------------
if __name__ == "__main__":
    run()


#--------------------- End of Script ---------------------
#src/data_preprocessing.py