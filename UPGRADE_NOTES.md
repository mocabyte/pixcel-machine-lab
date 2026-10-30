# Upgrade notes — October 2026

All scripts were tested against the library versions current in October 2026
(Python 3.13, mediapipe 1.1, OpenCV 5.0, Ultralytics 8.4, supervision 0.31,
open3d 0.20, torch 2.14).

## What broke upstream

| Library | Change | Affected script |
|---|---|---|
| MediaPipe 1.x | `mp.solutions` (Hands, FaceMesh, drawing_utils) removed. Only the Tasks API remains. | `gesture_recognition_basics.py` |
| OpenCV 5 | Haar cascade XML files no longer ship with the pip package (`cv2.data.haarcascades` is empty). | both face detection scripts |
| supervision 0.31 | `sv.ByteTrack` removed (moved to the separate `trackers` package). | `creative-object-tracker.py` |
| Ultralytics 8.4 | ByteTrack needs `lap`, which is no longer reliably auto-installed. New default family: YOLO26. | tracker, YOLO challenge |

## What changed here

- `src/common.py` (new): repo-relative paths, model download cache in `models/`,
  device selection (CUDA / Apple Silicon MPS / CPU). Scripts now run from any folder.
- `data/haarcascade_frontalface_default.xml` (new): copied from the OpenCV repository.
- `gesture_recognition_basics.py`: rewritten for the Tasks API. Uses the Gesture
  Recognizer, so it now actually recognizes gestures (Thumb_Up, Victory, Open_Palm …),
  plus the Face Landmarker. Models download on first run. Accepts a video path.
- `creative-object-tracker.py`: one tracker instead of two. IDs from `model.track()`
  reach the labels (`#3 cat 0.91`), so tracking is visibly different from detection.
  Pressing q no longer kills the script before the heatmap.
- `face_detection.py`: runs at all — real image paths, frontal-face cascade (was the
  smile cascade), loops over all three images, saves results.
- `face_detection_videostream.py`: no crash at video end; accepts a video path.
- `cv_numpy_basics_images.py`: red channel really is red now (RGB index 0), plus a
  short BGR-vs-RGB demo with `cv2.imread`.
- `yolo_challenge.py`: includes the `anti_vision_*` images, YOLO26 default
  (`yolov8n.pt` one line away), saves with OpenCV, `imageio` no longer needed.
- `gan_fake_digits.py`: "Alle Ziffern" / "bestimmte Ziffer" no longer swapped; digit is
  a variable; GPU/MPS support; per-epoch loss print; output in `output/gan/`.
- `mini_digital_twin.py`: global registration (FPFH + RANSAC) before ICP. Default runs
  in demo mode on `elephant.ply` with a simulated second scan. ICP alone from identity:
  fitness 0.00. Global + ICP: 0.98.
- `loading_displaying_saving_image.py`: runnable example at the bottom.
- `requirements.txt`: tested versions with upper bounds; `.gitignore` for models/output.

## Not verified in the test environment

- MediaPipe inference itself: the `.task` model downloads were blocked in the sandbox.
  API, drawing and timestamps were tested with synthetic landmarks. Run it once on a laptop.
- MNIST download (same reason); the GAN loop was tested on placeholder data.
- Live webcam input and GUI windows (headless test harness).
