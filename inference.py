import joblib
from extract_features import extract_features

clf = joblib.load("model.pkl")

def predict(image_path):
    
    feat, mode, att = extract_features(image_path)

    if feat is None:
        return "Face bulunamadı!", "ERROR", None

    prob = clf.predict_proba(feat)[0]

    real_prob = prob[0]
    fake_prob = prob[1]

    if fake_prob > 0.52:
        result = f"FAKE (%{fake_prob*100:.2f})"
    else:
        result = f"REAL (%{real_prob*100:.2f})"

    return result, mode, att