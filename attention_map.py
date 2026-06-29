import numpy as np
import cv2

def create_heatmap(image_path, attention):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    attention = attention.numpy()
    attention = attention.reshape(14, 14)  # 16x16 patch → 14x14 approx

    attention = cv2.resize(attention, (224, 224))

    attention = (attention - attention.min()) / (attention.max() + 1e-8)

    heatmap = cv2.applyColorMap(
        np.uint8(255 * attention),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return overlay