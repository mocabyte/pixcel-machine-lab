"""
creative-object-tracker.py  — OOP Tracking Template
Object Tracking with YOLO + ByteTrack (Ultralytics) + Supervision annotators

Features
- OOP-Struktur (ModelManager, TrackerPipeline)
- YOLO-Detektion + ByteTrack-Tracking, EIN Tracker (der von Ultralytics)
- Labels mit Track-ID: "#3 cat 0.91" — die ID ist das, was Tracking von Detektion unterscheidet
- Bewegungs-Heatmap (mit Blur & Overlay auf letztem Frame)
- Kreative Erweiterungs-Stubs (Confidence Pulse, Motion Vectors, Memory Trail, AI Narrator)

Hinweis: supervision hat sv.ByteTrack entfernt (jetzt im separaten Paket
`trackers`). Wir brauchen es nicht — model.track() trackt bereits.

Install
  pip install -r requirements.txt
Run
  python src/creative-object-tracker.py
"""

from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

from common import data_path, get_device


# ----------------------------------------------------------
# 1) ModelManager — lädt und erklärt YOLOv8-Varianten
# ----------------------------------------------------------
class ModelManager:
    """
    Verwaltet das Laden verschiedener YOLOv8-Modelle und erklärt den Trade-off.

    YOLO model zoo (Größe → Genauigkeit vs. Geschwindigkeit), gleich für yolov8 und yolo26:
      - n ("nano")     🐇  sehr schnell, geringste Genauigkeit; gut für Echtzeit-Demos
      - s ("small")    ⚖️  guter Mittelweg; oft ideal für Laptops ohne starke GPU
      - m ("medium")   🚗  besser bei kleineren/weiteren Objekten; moderat langsamer
      - l ("large")    🦁  hohe Genauigkeit, spürbar langsamer; bessere Crowd-Szenen
      - x ("x-large")  🐢  beste Genauigkeit, sehr rechenintensiv; für anspruchsvolle Szenen

    Merksatz:
      Größer = mehr Parameter/Layer → erkennt kleinste/weit entfernte Objekte stabiler,
      kostet aber FPS (gerade ohne starke GPU).
    """

    def __init__(self, model_size: str = "n", family: str = "yolo26"):
        """
        Args:
            model_size: 'n' | 's' | 'm' | 'l' | 'x'
            family: 'yolo26' (aktuell) oder 'yolov8' (Vergleich mit den Folien)
        """
        self.model_path = f"{family}{model_size}.pt"
        self.model = YOLO(self.model_path)
        print(f"✅ Loaded YOLO model: {self.model_path}")

    def get_model(self) -> YOLO:
        """Gibt das geladene YOLO-Objekt zurück."""
        return self.model


# ----------------------------------------------------------
# 2) TrackerPipeline — Detection, Tracking, Visualisierung & Kreativ-Hooks
# ----------------------------------------------------------
class TrackerPipeline:
    """
    Orchestriert Detektion (YOLO), Assoziation (ByteTrack) und Visualisierung (Supervision).
    Enthält fertige Heatmap-Implementierung + leere Methoden für kreative Erweiterungen.

    Workflow pro Frame:
      1) YOLO liefert Detections (xyxy, Klasse, Confidence)
      2) ByteTrack verknüpft Detections über Zeit zu stabilen IDs
      3) Box- & Label-Rendering (BoxAnnotator + LabelAnnotator)
      4) Creative Hooks (optional durch Studierende implementieren)
    """

    def __init__(self, model: YOLO, source_path: str):
        """
        Args:
            model: geladenes YOLO-Modell (aus ModelManager)
            source_path: Pfad zum Video
        """
        self.model = model
        self.source_path = source_path

        # supervision >= 0.26 trennt Box- und Label-Rendering:
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1)

        self.device = get_device()

        # Buffer für Auswertungen/Visualisierungen
        self.positions = []          # für Heatmap (Zentren)
        self.last_positions = {}     # tracker_id → letzter Mittelpunkt (Motion Vectors)
        self.trails = {}             # tracker_id → Liste letzter Mittelpunkte (Memory Trail)

        print("🚀 TrackerPipeline initialized")

    def process_video(self):
        """
        🎬 Hauptloop:
          - YOLO track (stream) → Detektion + ByteTrack-IDs in einem Schritt
          - YOLO-Result → sv.Detections (inkl. tracker_id)
          - Boxen + Labels zeichnen
          - Creative-Hooks anwenden
          - Am Ende Heatmap erzeugen
        """
        print(f"🎥 Processing video: {self.source_path}")
        frame_shape = None
        last_frame = None

        names = self.model.names

        for result in self.model.track(
            source=self.source_path, stream=True, tracker="bytetrack.yaml",
            persist=True, device=self.device, verbose=False,
        ):
            frame = result.orig_img
            last_frame = frame.copy()
            frame_shape = frame.shape

            # --- YOLO-Result → Supervision Detections (bringt tracker_id mit) ---
            tracked = sv.Detections.from_ultralytics(result)

            # Noch keine bestätigten Tracks in diesem Frame → trotzdem anzeigen
            if len(tracked) == 0 or tracked.tracker_id is None:
                if not self._display_frame(frame):
                    break
                continue

            # Labels mit ID, z. B. "#3 cat 0.91"
            labels = [
                f"#{tid} {names[int(cid)]} {conf:.2f}"
                for tid, cid, conf in zip(tracked.tracker_id, tracked.class_id, tracked.confidence)
            ]

            # 1) Boxen zeichnen
            annotated = self.box_annotator.annotate(scene=frame, detections=tracked)
            # 2) Text-Labels zeichnen
            annotated = self.label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)

            # Heatmap-Daten sammeln
            self._record_positions(tracked)

            # Kreative Erweiterungen (von Studierenden implementierbar)
            self._apply_confidence_pulse(annotated, tracked)
            self._draw_motion_vectors(annotated, tracked)
            self._tracker_memory_trail(annotated, tracked)
            self._ai_narrator(tracked)

            # Anzeige
            if not self._display_frame(annotated):
                break

        cv2.destroyAllWindows()

        # Heatmap am Ende (Overlay auf letztem Frame)
        if frame_shape is not None:
            self._generate_heatmap(frame_shape, last_frame=last_frame)

        print("✅ Video processing complete.")

    def _display_frame(self, frame) -> bool:
        """Zeigt das aktuelle Frame. Gibt False zurück, wenn 'q' gedrückt wurde."""
        cv2.imshow("Object Tracking", frame)
        return not (cv2.waitKey(1) & 0xFF == ord("q"))

    # ----------------------------------------------------------
    # CREATIVE 1 — Heatmap (implementiert, inkl. Blur+Overlay)
    # ----------------------------------------------------------
    def _record_positions(self, tracked: sv.Detections):
        """
        Speichert die Zentren aller Boxen → dient als Basis für die Heatmap.
        Bright = Bereiche mit hoher Bewegungsdichte.
        """
        for (x1, y1, x2, y2) in tracked.xyxy:
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            self.positions.append((x_center, y_center))

    def _generate_heatmap(self, frame_shape, last_frame=None):
        """
        Erzeugt eine gut sichtbare Bewegungs-Heatmap:
          - Verstärkt Signale (+= 20)
          - Glättet mit Gaussian Blur (51x51)
          - Normiert und färbt ein (COLORMAP_JET)
          - Legt sie optional halbtransparent über das letzte Videoframe
        """
        if not self.positions:
            print("⚠️ No positions recorded — run process_video() first.")
            return

        print("🔥 Generating enhanced movement heatmap...")
        heat = np.zeros(frame_shape[:2], dtype=np.float32)

        # Verstärkung — bei schwacher Bewegung ggf. 50/100 testen
        for (x, y) in self.positions:
            if 0 <= y < heat.shape[0] and 0 <= x < heat.shape[1]:
                heat[y, x] += 20

        heat = cv2.GaussianBlur(heat, (51, 51), 0)
        heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
        heat_color = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(last_frame, 0.5, heat_color, 0.7, 0) if last_frame is not None else heat_color
        cv2.imshow("Tracking Heatmap", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("✅ Heatmap visualization complete.")

    # ----------------------------------------------------------
    # CREATIVE 2 — Confidence Pulse (Stub)
    # ----------------------------------------------------------
    def _apply_confidence_pulse(self, frame, tracked: sv.Detections):
        """
        TODO: Visualisiere Unsicherheit.
        Ideen:
          - Linienbreite / Alpha abhängig von tracked.confidence
          - Unsichere Detections mit pulsierendem Rand
        """
        pass

    # ----------------------------------------------------------
    # CREATIVE 3 — Motion Vectors (Stub)
    # ----------------------------------------------------------
    def _draw_motion_vectors(self, frame, tracked: sv.Detections):
        """
        TODO: Bewegungsrichtungen/geschwindigkeit visualisieren.
        Schritte:
          - für jede ID (tracked.tracker_id) aktuellen Mittelpunkt berechnen
          - letzten Mittelpunkt aus self.last_positions nehmen
          - Pfeil (cv2.arrowedLine) alt → neu zeichnen; Farbe je nach Geschwindigkeit
        """
        pass

    # ----------------------------------------------------------
    # CREATIVE 4 — Memory Trail (Stub)
    # ----------------------------------------------------------
    def _tracker_memory_trail(self, frame, tracked: sv.Detections):
        """
        TODO: "Ghost Trails" der letzten N Positionen pro ID.
        Idee:
          - pro ID Liste letzter K Mittelpunkte speichern
          - Punkte mit abnehmender Intensität zeichnen (älter = transparenter)
        """
        pass

    # ----------------------------------------------------------
    # CREATIVE 5 — AI Narrator (Stub)
    # ----------------------------------------------------------
    def _ai_narrator(self, tracked: sv.Detections):
        """
        TODO: Kleine Textzusammenfassung generieren (alle N Frames).
        Beispiele:
          - Anzahl & Klassen zählen, in Konsole ausgeben
          - optional: Gen-AI anbinden für natürliches Kommentar
        """
        pass


# ----------------------------------------------------------
# 3) Main — Einstiegspunkt
# ----------------------------------------------------------
def main():
    # Model wählen: 'n' | 's' | 'm' | 'l' | 'x'
    model_manager = ModelManager(model_size="n")  # ändere z. B. auf 'x' um Unterschiede zu demonstrieren

    # Videoquelle anpassen (liegt in data/): cat.mp4, tauben.mp4, shibuya_crossing.MP4
    source_video = str(data_path("cat.mp4"))

    pipeline = TrackerPipeline(model=model_manager.get_model(), source_path=source_video)
    pipeline.process_video()


if __name__ == "__main__":
    main()
