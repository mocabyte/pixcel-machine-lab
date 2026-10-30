"""
MediaPipe Challenge — Hide or Haunt
-----------------------------------
Two missions, one rule: change ONE thing per attempt.

  Hide:  make MediaPipe miss a face or hand that is clearly there.
  Haunt: make MediaPipe find a face or hand that is not there.

The script runs the Face Landmarker and the Gesture Recognizer on an image,
a folder of images, a video or the webcam. It draws what the models see and
logs every frame to output/mediapipe_challenge.csv, so attempts can be
compared with numbers instead of impressions.

Examples:
  python src/mediapipe_challenge.py                                   # webcam
  python src/mediapipe_challenge.py --label "black gloves"            # webcam, named attempt
  python src/mediapipe_challenge.py data/anti_vision_makeup.png       # one image
  python src/mediapipe_challenge.py data/                             # every image in a folder
  python src/mediapipe_challenge.py "data/sign interpreter.mov"       # video
  python src/mediapipe_challenge.py "data/sign interpreter.mov" --mode image

--mode image vs. --mode video
  VIDEO mode detects once, then tracks from frame to frame.
  IMAGE mode starts from zero on every frame — no memory, no tracking.
  Same input, two modes: how much of the robustness is the tracking?

Keys (webcam / video):  ESC = quit   SPACE = pause
"""

import argparse
import csv
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

from common import download_model, output_path

GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CSV_PATH = output_path("mediapipe_challenge.csv")
CSV_FIELDS = [
    "run", "label", "source", "mode", "frame", "time_ms",
    "face_found", "smile",
    "hands", "gesture_1", "score_1", "gesture_2", "score_2",
]


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

class Models:
    """Face Landmarker + Gesture Recognizer in IMAGE or VIDEO running mode."""

    def __init__(self, mode: str, min_confidence: float):
        running = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        self.mode = mode

        self.faces = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(download_model(FACE_MODEL_URL))),
                running_mode=running,
                num_faces=1,
                min_face_detection_confidence=min_confidence,
                min_face_presence_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
                output_face_blendshapes=True,     # expressions, e.g. smile
            )
        )
        self.gestures = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=str(download_model(GESTURE_MODEL_URL))),
                running_mode=running,
                num_hands=2,
                min_hand_detection_confidence=min_confidence,
                min_hand_presence_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
        )

    def run(self, frame_bgr, time_ms: int):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.mode == "video":
            return (self.faces.detect_for_video(image, time_ms),
                    self.gestures.recognize_for_video(image, time_ms))
        return self.faces.detect(image), self.gestures.recognize(image)

    def close(self):
        self.faces.close()
        self.gestures.close()


# ---------------------------------------------------------
# Reading results
# ---------------------------------------------------------

def smile_score(face_result):
    """Mean of the two mouth-smile blendshapes, or None without a face."""
    if not face_result.face_blendshapes:
        return None
    scores = {c.category_name: c.score for c in face_result.face_blendshapes[0]}
    return (scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)) / 2


def hand_readings(gesture_result, mirrored: bool):
    """One (side, gesture, score) tuple per detected hand."""
    readings = []
    for i in range(len(gesture_result.hand_landmarks)):
        side = gesture_result.handedness[i][0].category_name if gesture_result.handedness else "?"
        if mirrored:
            side = {"Left": "Right", "Right": "Left"}.get(side, side)
        if gesture_result.gestures and gesture_result.gestures[i]:
            top = gesture_result.gestures[i][0]
            readings.append((side, top.category_name, top.score))
        else:
            readings.append((side, "None", 0.0))
    return readings


# ---------------------------------------------------------
# Drawing
# ---------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREEN, RED = (255, 255, 255), (80, 220, 80), (60, 60, 230)


def draw(frame, face_result, gesture_result, readings, smile, mode, label):
    h, w, _ = frame.shape
    scale = max(0.7, w / 1100)              # readable on webcam frames and big photos
    thick = max(2, round(scale * 2))

    for landmarks in face_result.face_landmarks:
        drawing_utils.draw_landmarks(
            frame, landmarks,
            vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )

    for landmarks, (side, gesture, score) in zip(gesture_result.hand_landmarks, readings):
        drawing_utils.draw_landmarks(
            frame, landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )
        wrist = landmarks[0]
        pos = (int(wrist.x * w), min(h - 10, int(wrist.y * h + 40 * scale)))
        text = f"{side}: {gesture} {score:.2f}"
        cv2.putText(frame, text, pos, FONT, scale, (0, 0, 0), thick + 2)
        cv2.putText(frame, text, pos, FONT, scale, WHITE, thick)

    # status panel, top left
    face_txt = f"face: yes  smile {smile:.2f}" if smile is not None else "face: no"
    lines = [
        (face_txt, GREEN if smile is not None else RED),
        (f"hands: {len(readings)}", GREEN if readings else RED),
        (f"{mode.upper()} mode" + (f"  |  {label}" if label else ""), WHITE),
    ]
    for i, (text, color) in enumerate(lines):
        y = int((30 + i * 32) * scale)
        cv2.putText(frame, text, (int(12 * scale), y), FONT, scale, (0, 0, 0), thick + 2)
        cv2.putText(frame, text, (int(12 * scale), y), FONT, scale, color, thick)
    return frame


# ---------------------------------------------------------
# Logging and summary
# ---------------------------------------------------------

class Log:
    """Appends one CSV row per frame and keeps numbers for the summary."""

    def __init__(self, label: str):
        self.run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.label = label
        new_file = not CSV_PATH.exists()
        self.file = open(CSV_PATH, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        if new_file:
            self.writer.writeheader()
        self.stats = defaultdict(lambda: {"frames": 0, "face": [], "hands": 0,
                                          "gestures": Counter(), "scores": defaultdict(list)})

    def add(self, source, mode, frame_idx, time_ms, smile, readings):
        row = {
            "run": self.run, "label": self.label, "source": source, "mode": mode,
            "frame": frame_idx, "time_ms": time_ms,
            "face_found": int(smile is not None),
            "smile": "" if smile is None else f"{smile:.3f}",
            "hands": len(readings),
        }
        for i, (_, gesture, score) in enumerate(readings[:2], start=1):
            row[f"gesture_{i}"] = gesture
            row[f"score_{i}"] = f"{score:.3f}"
        self.writer.writerow(row)

        s = self.stats[source]
        s["frames"] += 1
        s["face"].append(smile is not None)
        s["hands"] += bool(readings)
        for _, gesture, score in readings:
            s["gestures"][gesture] += 1
            s["scores"][gesture].append(score)

    def summary(self):
        self.file.close()
        print("\n" + "=" * 60)
        print(f"Attempt: {self.label or '(no label)'}")
        for source, s in self.stats.items():
            n = s["frames"]
            print(f"\n{source}  ({n} frame{'s' if n != 1 else ''})")
            print(f"  face found : {sum(s['face']) / n:6.1%}")
            print(f"  hand found : {s['hands'] / n:6.1%}")
            if n > 1:
                print(f"  longest face loss: {longest_gap(s['face'])} frames")
            for gesture, count in s["gestures"].most_common():
                mean = sum(s["scores"][gesture]) / len(s["scores"][gesture])
                print(f"  {gesture:<12} {count:5d}x   mean confidence {mean:.2f}")
        print(f"\nLog: {CSV_PATH}")


def longest_gap(found):
    """Longest run of frames without a face, counted after the first detection."""
    best = current = 0
    seen = False
    for f in found:
        if f:
            seen, current = True, 0
        elif seen:
            current += 1
            best = max(best, current)
    return best


# ---------------------------------------------------------
# Sources
# ---------------------------------------------------------

def run_images(paths, models, log, label, show):
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Skipped (not readable): {path.name}")
            continue
        face_result, gesture_result = models.run(frame, 0)
        readings = hand_readings(gesture_result, mirrored=False)
        smile = smile_score(face_result)
        log.add(path.name, models.mode, 0, 0, smile, readings)

        annotated = draw(frame, face_result, gesture_result, readings, smile, models.mode, label)
        cv2.imwrite(str(output_path(f"challenge_{path.stem}.jpg")), annotated)
        if show:
            cv2.imshow("MediaPipe Challenge", annotated)
            if cv2.waitKey(0) & 0xFF == 27:
                break


def run_stream(source, models, log, label, show):
    webcam = source == 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {source}")
    name = "webcam" if webcam else Path(source).name
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    start = time.monotonic()
    frame_idx, last_ms = 0, -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if webcam:
            frame = cv2.flip(frame, 1)
            time_ms = int((time.monotonic() - start) * 1000)
        else:
            time_ms = int(frame_idx * 1000 / fps)
        time_ms = max(time_ms, last_ms + 1)     # VIDEO mode needs rising timestamps
        last_ms = time_ms

        face_result, gesture_result = models.run(frame, time_ms)
        readings = hand_readings(gesture_result, mirrored=webcam)
        smile = smile_score(face_result)
        log.add(name, models.mode, frame_idx, time_ms, smile, readings)
        frame_idx += 1

        if show:
            cv2.imshow("MediaPipe Challenge",
                       draw(frame, face_result, gesture_result, readings, smile, models.mode, label))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32:                       # SPACE: pause until any key
                cv2.waitKey(0)

    cap.release()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hide or Haunt: challenge MediaPipe.")
    parser.add_argument("source", nargs="?", default="0",
                        help="image, folder, video, or 0 for the webcam (default)")
    parser.add_argument("--label", default="", help='name of this attempt, e.g. "black gloves"')
    parser.add_argument("--mode", choices=["auto", "image", "video"], default="auto",
                        help="auto = image for photos, video for streams")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="minimum confidence for detection and tracking (0–1)")
    parser.add_argument("--no-window", action="store_true", help="log only, no display")
    args = parser.parse_args()

    if args.source == "0":
        source, images = 0, None
    else:
        path = Path(args.source)
        if path.is_dir():
            images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_TYPES)
        elif path.suffix.lower() in IMAGE_TYPES:
            images = [path]
        else:
            images = None
        source = str(path)

    if images is not None and args.mode == "video":
        parser.error("--mode video needs a video or the webcam; photos have no time axis")
    mode = args.mode if args.mode != "auto" else ("image" if images is not None else "video")

    models = Models(mode, args.confidence)
    log = Log(args.label)
    try:
        if images is not None:
            run_images(images, models, log, args.label, show=not args.no_window)
        else:
            run_stream(source, models, log, args.label, show=not args.no_window)
    except KeyboardInterrupt:
        pass
    finally:
        models.close()
        cv2.destroyAllWindows()
        log.summary()


if __name__ == "__main__":
    main()
