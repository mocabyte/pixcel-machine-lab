# YOLO Challenge — What Does the Machine Really See?
#
# Part 1: everyday images (dog, horses) — YOLO's comfort zone.
# Part 2: art and anti-vision images — fashion, hair and make-up designed
#         to confuse detectors. Compare what you see with what YOLO sees.

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from common import data_path, output_path, get_device

# --- Settings -----------------------------------------------------
MODEL = "yolo26n.pt"   # current generation; try "yolov8n.pt" to compare with the slides,
                       # or bigger sizes: s / m / l / x
CONF = 0.25            # confidence threshold — lower it and watch what appears
SAVE_RESULTS = True    # save annotated images into output/

# Images to test (all in data/)
IMAGES = [
    "dog.jpg",
    "horses.jpg",
    "scream.jpg",
    "anti_vision_fashion.jpg",
    "anti_vision_hair.jpg",
    "anti_vision_makeup.png",
]

# --- Run YOLO -----------------------------------------------------
device = get_device()
print(f"Loading model: {MODEL} on {device}")
model = YOLO(MODEL)

for name in IMAGES:
    img_path = data_path(name)
    print(f"\nProcessing: {img_path.name}")
    result = model(img_path, conf=CONF, device=device, verbose=False)[0]  # list → first item

    # --- Print detections in terminal
    if len(result.boxes) == 0:
        print("No detections found.")
    for box in result.boxes:
        cls = result.names[int(box.cls)]
        conf = float(box.conf)
        print(f"→ {cls} ({conf:.2f})")

    # --- Annotated image (YOLO draws bounding boxes)
    annotated_bgr = result.plot()                # BGR format (OpenCV style)
    annotated_rgb = annotated_bgr[:, :, ::-1]    # convert to RGB for Matplotlib

    # --- Show the image
    plt.figure(figsize=(8, 6))
    plt.imshow(annotated_rgb)
    plt.title(f"Detections for: {img_path.name}")
    plt.axis("off")
    plt.show(block=False)
    plt.pause(5)   # keep open briefly when running in scripts
    plt.close()

    # --- Optional: save annotated image (cv2 expects BGR)
    if SAVE_RESULTS:
        save_path = output_path(f"{img_path.stem}_detected.jpg")
        cv2.imwrite(str(save_path), annotated_bgr)
        print(f"Saved: {save_path}")

print("\n✅ YOLO Challenge complete!")
print("Now reflect: What does the machine really see — and what does it miss?")
