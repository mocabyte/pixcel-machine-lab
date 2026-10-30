"""
Face Detection with Haar Cascades (Viola–Jones, 2001)

Classic, pre-deep-learning face detection: hand-designed features,
a cascade of simple classifiers, a sliding window over a grayscale image.

Note: OpenCV 5 no longer ships the cascade XML files inside the pip
package, so the file lives in data/ now.

Run:
  python src/face_detection.py
"""

import cv2

from common import data_path, output_path

# Images (all in data/) — try each one and compare
IMAGES = {
    "sabalenka": "Aryna_Sabalenka.jpg",
    "mukwege": "Denis-Mukwege.jpg",
    "solvay": "1911_Solvay_conference.jpg",   # 29 faces, small and dense
}

# Cascade file: the frontal face model
face_cascade = cv2.CascadeClassifier(str(data_path("haarcascade_frontalface_default.xml")))
if face_cascade.empty():
    raise RuntimeError("Cascade could not be loaded.")


def detect_face(img_gray, scale_factor=1.3, min_neighbors=5):
    """Return a copy of the image with a white box around every face found."""
    face_img = img_gray.copy()
    face_rects = face_cascade.detectMultiScale(
        face_img, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 3)
    return face_img, len(face_rects)


for name, filename in IMAGES.items():
    # Haar cascades work on grayscale: flag 0 = cv2.IMREAD_GRAYSCALE
    img = cv2.imread(str(data_path(filename)), cv2.IMREAD_GRAYSCALE)

    # Experiment: lower scaleFactor (e.g. 1.1) and minNeighbors for the Solvay photo
    result, count = detect_face(img)
    print(f"{name}: {count} face(s) found")

    cv2.imwrite(str(output_path(f"faces_{name}.jpg")), result)

    # Show image in OpenCV window, any key → next image
    cv2.imshow(f"Detected Faces – {name}", result)
    cv2.waitKey(0)

cv2.destroyAllWindows()
