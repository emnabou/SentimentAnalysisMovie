import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# Load the preprocessed data
X, y = joblib.load("processed_Movie/preprocessed_data.pkl")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("[INFO] Data split into training and testing sets.")
print(f"[INFO] Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")


# Choose a model (You can switch between Naive Bayes or Logistic Regression)
model = MultinomialNB()  # Uncomment to use Naive Bayes
#model = LogisticRegression(max_iter=200) # Uncomment to use Logistic Regression

# Train the model
model.fit(X_train, y_train)
model.fit(X,y)
print("[INFO] Model trained successfully.")

#Evaluate the model
print("[INFO] Evaluating the model....")
y_pred = model.predict(X_test)

#Metrics

accuracy=accuracy_score(y_test, y_pred)
report=classification_report(y_test, y_pred, target_names=["Negative", "Positive"])

print(f"[RESULT] Accuracy: {accuracy:.2f}")
print("[RESULT] Classification Report:\n", report)

#save the trained model
os.makedirs("models",exist_ok=True)
joblib.dump(model, "models/sentiment_classifier_movie.pkl")
print("[INFO] Model saved successfully.")
#src/train_model.py