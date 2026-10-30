import os
import sys

import cv2

from common import data_path


# Bild laden
def loadImage(filename):
    img = cv2.imread(str(filename))
    if img is None:
        raise FileNotFoundError(f"Bild konnte nicht geladen werden: {filename}")
    return img


# Bild anzeigen
def displayImage(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Bild speichern
def saveImage(img, filename):
    # Pfad aufteilen
    name, extension = os.path.splitext(str(filename))
    # neuen Dateinamen generieren
    new_filename = f"{name}_copy{extension}"

    # Bild speichern
    cv2.imwrite(new_filename, img)
    return new_filename


if __name__ == "__main__":
    # Eigenes Bild: python src/loading_displaying_saving_image.py pfad/zum/bild.jpg
    path = sys.argv[1] if len(sys.argv) > 1 else data_path("red_panda.jpg")
    img = loadImage(path)
    print(f"Geladen: {path} — Form (H, B, Kanäle): {img.shape}")
    displayImage(img)
    print(f"Gespeichert: {saveImage(img, path)}")
