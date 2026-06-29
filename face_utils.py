from mtcnn import MTCNN
import cv2

detector = MTCNN()

def get_face(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None, "ERROR"

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb)

    # -------------------------
    # FACE FOUND
    # -------------------------
    if len(faces) > 0:

        x, y, w, h = faces[0]['box']

        x, y = max(0, x), max(0, y)

        face = rgb[y:y+h, x:x+w]

        face = cv2.resize(face, (224, 224))

        return face, "FACE"

    # -------------------------
    # NO FACE → GLOBAL IMAGE
    # -------------------------
    else:

        global_img = cv2.resize(rgb, (224, 224))

        return global_img, "GLOBAL"