import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from face_utils import get_face

model_name = "google/vit-base-patch16-224"

processor = AutoImageProcessor.from_pretrained(model_name)

model = AutoModel.from_pretrained(
    model_name,
    output_attentions=True
)

model.eval()


def extract_features(image_path):

    img, mode = get_face(image_path)

    if img is None:
        return None, None, None

    image = Image.fromarray(img)

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # CLS token feature
    features = outputs.last_hidden_state[:, 0, :].numpy()

    # Attention (son layer)
    attentions = outputs.attentions[-1]  # (1, heads, tokens, tokens)

    attention_map = attentions.mean(dim=1)[0]  # head avg

    cls_attention = attention_map[0, 1:]  # CLS → patch attention

    return features, mode, cls_attention