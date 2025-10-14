"""
AI & Computer Vision – Practical Activity
-----------------------------------------
Hands + FaceMesh Tracking Demo (stable + educational)

Demonstriert:
- wie Deep Learning Modelle (MediaPipe) Landmarken in Echtzeit erkennen,
- die visuelle Hierarchie von "Pixel → Features → Bedeutung",
- OOP-Struktur, um visuelle Pipelines sauber zu kapseln.

Getestet mit:
Python 3.10–3.11
mediapipe 0.10.14
opencv-python 4.10
"""

import cv2
import mediapipe as mp


class VisionTracker:
    """Kombiniert MediaPipe Hands und FaceMesh in einer stabilen OpenCV-Pipeline."""

    def __init__(self, camera_index: int = 0):
        # Drawing-Utilities & Styles (Farbdefinitionen für Landmarken)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # MediaPipe Modelle
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh

        # Kamera öffnen
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Kamera konnte nicht geöffnet werden.")

        # Hand-Tracking-Modell (21 Landmarken)
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # FaceMesh-Modell (468 Punkte + Iris)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    # -----------------------------------------------------
    # Hilfsfunktionen für das Zeichnen
    # -----------------------------------------------------

    def _draw_hands(self, image, results):
        """Zeichnet Hand-Landmarks mit Standard-Styles."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )

    def _draw_face_mesh(self, image, results):
        """Zeichnet Gesichts-Landmarks stabil (mit Fallback bei MediaPipe-Bug)."""
        if not results.multi_face_landmarks:
            return

        for face_landmarks in results.multi_face_landmarks:
            try:
                # Primärer Versuch mit offiziellen Styles
                self.mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    self.mp_styles.get_default_face_mesh_contours_style()
                )
            except KeyError:
                # Bekannter Bug-Fix: fallback – Punkte manuell zeichnen
                print("⚠️  FaceMesh-Drawing-Spec fehlte – Fallback aktiviert.")
                h, w, _ = image.shape
                for lm in face_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 1, (0, 255, 0), -1)

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
                    print("⚠️  Kein Kamerabild empfangen – beende Programm.")
                    break

                # Spiegelung für intuitive Darstellung
                frame = cv2.flip(frame, 1)

                # MediaPipe erwartet RGB-Bilder
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Deep-Learning-Inference (Landmark Detection)
                hand_results = self.hands.process(rgb)
                face_results = self.face_mesh.process(rgb)

                # Ergebnisse zeichnen
                self._draw_hands(frame, hand_results)
                self._draw_face_mesh(frame, face_results)

                # Anzeige
                cv2.imshow("AI & Computer Vision – Hands + FaceMesh", frame)

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
        self.hands.close()
        self.face_mesh.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Ressourcen freigegeben. Session beendet.")


# ---------------------------------------------------------
# Programmlogik
# ---------------------------------------------------------
if __name__ == "__main__":
    tracker = VisionTracker()
    tracker.run()
