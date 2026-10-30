# Pixcel Machine Lab

Learning and experimenting with AI + Computer Vision.

## Structure
- `src/` → Python scripts (Haar, ResNet, YOLO)
- `notebooks/` → Jupyter notebooks for step-by-step explanations (maybe)
- `data/` → sample images & videos
- `requirements.txt` → dependencies

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Run every script from anywhere, e.g. `python src/yolo_challenge.py`.
Paths resolve via `src/common.py`; models are downloaded into `models/` on first use,
results land in `output/`. Tested October 2026 on Python 3.13.

| Script | What it shows | Input |
|---|---|---|
| `loading_displaying_saving_image.py` | read, show, write an image | `data/red_panda.jpg` or your own |
| `cv_numpy_basics_images.py` | images are arrays; RGB vs BGR | `data/red_panda.jpg` |
| `face_detection.py` | Haar cascades on stills | three portraits in `data/` |
| `face_detection_videostream.py` | Haar cascades on video | webcam or a video path |
| `yolo_challenge.py` | YOLO on everyday vs. anti-vision images | `data/` |
| `creative-object-tracker.py` | tracking = detection + stable IDs | `data/cat.mp4` |
| `gesture_recognition_basics.py` | hand landmarks → gestures, face landmarks | webcam or a video path |
| `gan_fake_digits.py` | a GAN learns to draw digits | MNIST (auto-download) |
| `mini_digital_twin.py` | two scans → aligned, merged, meshed, compared | `data/elephant.ply` |

## Goals
- E.g. Compare traditional (Haar Cascades) and modern (CNN, YOLOv8) face detection
- Provide reproducible experiments for media informatics students
- Serve as a living lab for computer vision exploration

