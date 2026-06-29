from PIL import Image, ImageChops, ImageEnhance

def generate_ela(image_path):

    original = Image.open(image_path).convert("RGB")

    temp_path = "temp_ela.jpg"

    original.save(temp_path, "JPEG", quality=90)

    compressed = Image.open(temp_path)

    diff = ImageChops.difference(original, compressed)

    extrema = diff.getextrema()

    max_diff = max([ex[1] for ex in extrema])

    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(diff).enhance(scale)

    return ela_image