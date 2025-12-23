import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "isl_model.pkl"))
labels = np.load(os.path.join(BASE_DIR, "labels.npy"), allow_pickle=True)

def recognize_gesture(landmarks):
    prediction = model.predict([landmarks])[0]
    return labels[prediction]
