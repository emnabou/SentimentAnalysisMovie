# Movie Review Sentiment Analysis

This project is a **Sentiment Analysis** application built to predict whether a movie review is **Positive** or **Negative**. It uses a machine learning model trained on a processed movie review dataset.

## Features

- **Text Preprocessing**: The reviews are cleaned (removing punctuation, special characters, and converting to lowercase) before they are vectorized.
- **Machine Learning**: The project uses a **Logistic Regression** or **MultinomialNB** classifier to predict sentiment.
- **TF-IDF Vectorization**: The text data is vectorized using **TF-IDF** to convert text into numerical features.
- **Prediction**: After loading the model and vectorizer, the app can predict whether a given review is **Positive** or **Negative**.

## Prerequisites

Make sure you have the following installed:
- **Python 3.x**
- **pip** (Python package installer)

You will also need to install the required Python libraries, which can be done by running:

```bash
  pip install -r requirements.txt
```

## Dataset File
The dataset used in this project is: https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis?resource=download

Due to its large size I uploaded the dataset as a zip file that you only need to uzip and place it in your project directory\data or else change the file path to your csv file path

**Dataset Structure**

The dataset is in CSV format with two columns:

- **review**: The text of the movie review.
- **label**: Sentiment label (0 for Negative, 1 for Positive).

## Usage
1. Unzip the dataset
```bash
  unzip processed_movie_dataset.zip
```
2. Run the data_preprocessing.py script
   this will clean your dataset and vectorized using the TF-IDF Vectorizer and then store the preprocessed data in processed_Movie/preprocessed_data.pkl
   and the Vectorizer in "vectorizer.pkl"
   
```bash
  python data_preprocessing.py
```
3. Run the Training Script: If you want to train the model:
```bash
  python train_model.py
```
4. Run the Prediction Script: After training the model (or loading pre-trained model files), you can test predictions:
  ```bash
  python predict.py
```
You will be prompted to enter a movie review, and the app will predict whether the sentiment is Positive or Negative.
