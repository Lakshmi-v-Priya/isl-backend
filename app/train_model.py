import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

X = np.load("X.npy")
y = np.load("y.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
model.fit(X_train, y_train)

joblib.dump(model, "isl_model.pkl")
print("Model trained and saved!")
