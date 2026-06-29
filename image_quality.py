

import cv2
import numpy as np

def analyze_quality(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur seviyesi
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Parlaklık
    brightness = np.mean(gray)

    # Çözünürlük
    height, width = gray.shape

    return {
        "width": width,
        "height": height,
        "blur": round(blur_score, 2),
        "brightness": round(brightness, 2)
    }