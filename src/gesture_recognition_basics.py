"""
AI & Computer Vision – Practical Activity
-----------------------------------------
Gesture Recognition + Face Landmarks (MediaPipe Tasks API)

Demonstriert:
- wie Deep Learning Modelle (MediaPipe) Landmarken in Echtzeit erkennen,
- die visuelle Hierarchie "Pixel → Landmarken → Bedeutung (Geste)",
- OOP-Struktur, um visuelle Pipelines sauber zu kapseln.

Neu gegenüber der alten Version:
- Nutzt die MediaPipe Tasks API. Das alte `mp.solutions` (Hands, FaceMesh)
  gibt es in aktuellen MediaPipe-Versionen nicht mehr.
- Erkennt jetzt wirklich Gesten (Thumb_Up, Victory, Open_Palm, ...),
  nicht nur Landmarken.
- Modelle (.task) werden beim ersten Start automatisch in models/ geladen.

Getestet mit:
Python 3.10–3.13, mediapipe 1.1, opencv 5.0

Start:
  python src/gesture_recognition_basics.py            # Webcam
  python src/gesture_recognition_basics.py video.mp4  # Videodatei
"""

import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

from common import download_model

GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


class VisionTracker:
    """Kombiniert Gesture Recognizer (Hände) und Face Landmarker in einer OpenCV-Pipeline."""

    def __init__(self, source=0, num_hands: int = 2):
        # Kamera oder Video öffnen
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Quelle konnte nicht geöffnet werden: {source}")
        self.mirror = source == 0  # nur Webcam spiegeln

        # Gesture Recognizer: 21 Hand-Landmarken + Gesten-Klassifikation
        self.gestures = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=str(download_model(GESTURE_MODEL_URL))),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=num_hands,
                min_hand_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
        )

        # Face Landmarker: 478 Punkte (inkl. Iris)
        self.faces = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(download_model(FACE_MODEL_URL))),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
        )

        self.start = time.monotonic()
        self.last_ts = -1

    # -----------------------------------------------------
    # Hilfsfunktionen für das Zeichnen
    # -----------------------------------------------------

    def _draw_hands(self, image, result):
        """Zeichnet Hand-Landmarks und schreibt die erkannte Geste daneben."""
        h, w, _ = image.shape
        for i, landmarks in enumerate(result.hand_landmarks):
            drawing_utils.draw_landmarks(
                image,
                landmarks,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )

            # Bedeutung: Top-Geste dieser Hand (Kategorie + Score)
            if result.gestures and result.gestures[i]:
                top = result.gestures[i][0]
                side = result.handedness[i][0].category_name
                if self.mirror:  # Spiegelbild: links/rechts tauschen
                    side = {"Left": "Right", "Right": "Left"}.get(side, side)
                wrist = landmarks[0]
                pos = (int(wrist.x * w), int(wrist.y * h) + 30)
                text = f"{side}: {top.category_name} {top.score:.2f}"
                cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _draw_face(self, image, result):
        """Zeichnet die Gesichts-Konturen."""
        for landmarks in result.face_landmarks:
            drawing_utils.draw_landmarks(
                image,
                landmarks,
                vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
            )

    def _timestamp_ms(self) -> int:
        """VIDEO-Modus braucht streng steigende Zeitstempel."""
        ts = int((time.monotonic() - self.start) * 1000)
        ts = max(ts, self.last_ts + 1)
        self.last_ts = ts
        return ts

    # -----------------------------------------------------
    # Haupt-Loop
    # -----------------------------------------------------

    def run(self):
        """Startet Live-Tracking-Loop."""
        print("🎥 VisionTracker gestartet – ESC zum Beenden.")

        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    print("⚠️  Kein Bild mehr – beende Programm.")
                    break

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                # MediaPipe erwartet RGB, OpenCV liefert BGR
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = self._timestamp_ms()

                # Deep-Learning-Inference
                gesture_result = self.gestures.recognize_for_video(mp_image, ts)
                face_result = self.faces.detect_for_video(mp_image, ts)

                # Ergebnisse zeichnen
                self._draw_face(frame, face_result)
                self._draw_hands(frame, gesture_result)

                cv2.imshow("AI & Computer Vision – Gestures + Face", frame)

                # ESC-Taste (27) → Beenden
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        except KeyboardInterrupt:
            print("⏹️  Manuell unterbrochen.")
        finally:
            self.close()

    # -----------------------------------------------------
    # Ressourcen-Management
    # -----------------------------------------------------

    def close(self):
        """Schließt alle Ressourcen sauber."""
        self.gestures.close()
        self.faces.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Ressourcen freigegeben. Session beendet.")


# ---------------------------------------------------------
# Programmlogik
# ---------------------------------------------------------
if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    VisionTracker(source).run()
