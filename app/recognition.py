import joblib
import numpy as np

model = joblib.load("isl_model.pkl")
labels = np.load("labels.npy")

def recognize_gesture(landmarks):
    prediction = model.predict([landmarks])[0]
    return labels[prediction]
