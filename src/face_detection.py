import numpy as np
import cv2
import matplotlib.pyplot as plt

# Face Detection with Haar Cascades

# Images add your image path to 'image'
nadia = cv2.imread('image', 0)
denis = cv2.imread('image', 0)
solvay = cv2.imread('image', 0)

# Cascade File
face_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')

def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 3)
    return face_img

# Apply detection
result = detect_face(denis)

# Show image in OpenCV Frame
cv2.imshow("Detected Faces", result)

# Wait for key press and close
cv2.waitKey(0)
cv2.destroyAllWindows()
