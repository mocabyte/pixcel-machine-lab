# YOLOv8 Challenge — What Does the Machine Really See?

from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

# --- Settings -----------------------------------------------------
MODEL = "yolov8n.pt"  # or 'yolov8s.pt', 'yolov8m.pt' for comparison
IMAGE_DIR = Path("darknet/data")
SAVE_RESULTS = True    # save annotated images into "runs/" folder

# Images to test
IMAGES = [
    IMAGE_DIR / "dog.jpg",
    IMAGE_DIR / "horses.jpg",
    IMAGE_DIR / "scream.jpg"
]

# --- Run YOLO -----------------------------------------------------
print(f"Loading model: {MODEL}")
model = YOLO(MODEL)

for img_path in IMAGES:
    print(f"\nProcessing: {img_path.name}")
    result = model(img_path)[0]  # YOLO returns list; take first item

    # --- Print detections in terminal
    if not result.boxes:
        print("No detections found.")
        continue

    for box in result.boxes:
        cls = result.names[int(box.cls)]
        conf = float(box.conf)
        print(f"→ {cls} ({conf:.2f})")

    # --- Annotated image (YOLO draws bounding boxes)
    annotated_bgr = result.plot()            # BGR format (OpenCV style)
    annotated_rgb = annotated_bgr[:, :, ::-1]  # convert to RGB for Matplotlib

    # --- Show the image
    plt.figure(figsize=(8, 6))
    plt.imshow(annotated_rgb)
    plt.title(f"Detections for: {img_path.name}")
    plt.axis("off")
    plt.show(block=False)
    plt.pause(5)   # keep open briefly when running in scripts

    # --- Optional: save annotated image
    if SAVE_RESULTS:
        save_path = Path("runs") / f"{img_path.stem}_detected.jpg"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        from imageio.v2 import imwrite
        imwrite(save_path, annotated_rgb)
        print(f"Saved: {save_path}")

print("\n✅ YOLOv8 Challenge complete!")
print("Now reflect: What does the machine really see — and what does it miss?")
