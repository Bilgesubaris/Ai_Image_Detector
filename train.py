import os
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import matplotlib.pyplot as plt

from extract_features import extract_features

# ----------------------------
# DATASET
# ----------------------------
X = []
y = []

print("Feature extraction başlıyor...")

for label_name, label in [("real", 0), ("fake", 1)]:

    folder = f"dataset/train/{label_name}"

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        try:
            result = extract_features(path)

            # ❗ güvenlik kontrolü
            if result is None:
                continue

            feat, mode, att = result

            if feat is None:
                continue

            X.append(feat[0])  # CLS token flatten
            y.append(label)

        except Exception as e:
            print("Hata:", path, "->", e)
            continue

X = np.array(X)
y = np.array(y)

print("\nDataset Shape:", X.shape)

# ----------------------------
# CLASS BALANCE
# ----------------------------
print("\nClass Distribution:")
print("REAL:", sum(y == 0))
print("FAKE:", sum(y == 1))

# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# MODEL
# ----------------------------
clf = LogisticRegression(max_iter=50)
clf.fit(X_train, y_train)

# ----------------------------
# PREDICTION
# ----------------------------
pred = clf.predict(X_test)

# ----------------------------
# METRICS
# ----------------------------
print("\nAccuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(cm, display_labels=["REAL", "FAKE"])
disp.plot()

plt.title("Confusion Matrix")
plt.show()

# ----------------------------
# SAVE MODEL
# ----------------------------
joblib.dump(clf, "model.pkl")

print("\nMODEL KAYDEDİLDİ")