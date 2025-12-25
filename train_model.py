import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from app.landmark_extractor import extract_landmarks

# ===============================
# CONFIG
# ===============================
DATASET_DIR = "datasets"   # contains extracted JPG frames
MODEL_PATH = "app/isl_model.pkl"
LABELS_PATH = "app/labels.npy"

# ===============================
# INIT
# ===============================
X = []
y = []
label_names = []

print("🚀 TRAINING SCRIPT STARTED")
print("📂 Dataset directory:", DATASET_DIR)

# ===============================
# READ DATASET
# ===============================
for label_index, label in enumerate(os.listdir(DATASET_DIR)):
    label_path = os.path.join(DATASET_DIR, label)

    if not os.path.isdir(label_path):
        continue

    print(f"➡️ Processing label: {label}")
    label_names.append(label)

    for img_name in os.listdir(label_path):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        landmarks = extract_landmarks(image)

        if landmarks is None:
            # MediaPipe could not detect a hand
            continue

        X.append(landmarks)
        y.append(label_index)

# ===============================
# CHECK DATA
# ===============================
X = np.array(X)
y = np.array(y)

print("✅ Total samples collected:", len(X))

if len(X) == 0:
    raise RuntimeError(
        "❌ No hand landmarks detected.\n"
        "Check frame quality, lighting, and hand visibility."
    )

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# MODEL
# ===============================
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

print("🧠 Training model...")
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"📊 Model accuracy: {accuracy:.2f}")

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, MODEL_PATH)
np.save(LABELS_PATH, np.array(label_names))

print("🎉 TRAINING COMPLETE")
print("💾 Model saved to:", MODEL_PATH)
print("💾 Labels saved to:", LABELS_PATH)
