"""
Face Detection on a video stream (webcam or file) with Haar Cascades.

Run:
  python src/face_detection_videostream.py                 # webcam
  python src/face_detection_videostream.py data/cat.mp4    # video file
"""

import sys

import cv2

from common import data_path


def detect_faces(video_source):
    face_cascade = cv2.CascadeClassifier(str(data_path("haarcascade_frontalface_default.xml")))

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    while True:
        ret, frame = cap.read()
        if not ret:          # end of video or camera lost
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 0 = default webcam, or pass a video file path
    detect_faces(sys.argv[1] if len(sys.argv) > 1 else 0)
