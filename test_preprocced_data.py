import joblib

X, y = joblib.load("processed_Movie/preprocessed_data.pkl")

print("Feature Matrix Shape", X.shape)
print ("Label Shape", y.shape)
print("First 5 labels", y[:5])
#src/test_preprocced_data.py