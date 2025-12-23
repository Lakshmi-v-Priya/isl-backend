import os
import cv2
import numpy as np
from landmark_extractor import extract_landmarks

DATASET_PATH = "../datasets"
X, y = [], []
labels = os.listdir(DATASET_PATH)

label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)

        landmarks = extract_landmarks(img)
        if landmarks is not None:
            X.append(landmarks)
            y.append(label_map[label])

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)
np.save("labels.npy", labels)
