"""
Lab 02c — Measure Your Own Camera
Week 2 · Camera Geometry · Zhang (2000)

GOAL
  Find the four numbers that describe your own camera, then use them to
  measure something in the real world, then find where that measurement
  breaks.

  Nobody else in the room gets your numbers. That is the point.

WHAT YOU NEED
  A printed checkerboard. Any of the standard OpenCV patterns works.
  Count the INNER corners, not the squares: a board with 10 x 7 squares has
  9 x 6 inner corners. Measure one square with a ruler, in millimetres.

RUN
  Step 1 — collect images:
      python camera_calibration.py capture --out myboard

  Step 2 — solve for the camera:
      python camera_calibration.py solve --images myboard \\
             --cols 9 --rows 6 --square 25.0

  Step 3 — measure something with the result:
      python camera_calibration.py measure --calib myboard/camera.json \\
             --height-mm 297 --pixels 412

  Step 4 — see the distortion you just removed:
      python camera_calibration.py undistort --calib myboard/camera.json \\
             --image myboard/shot_03.jpg

CAPTURE KEYS
  space  keep this frame        q  stop capturing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import output_path


def open_source(source=0):
    """Open a webcam index or a video file and check that it opened."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source {source!r}. "
            "On macOS, grant camera access to your terminal under "
            "System Settings > Privacy & Security > Camera."
        )
    return cap


def frames(cap):
    """Yield frames until the stream ends."""
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            return
        yield frame


def banner(frame, lines, origin=(12, 12), scale=0.6):
    """Draw readable text on any background by shadowing it first."""
    x, y = origin
    line_h = int(28 * scale / 0.6)
    for i, line in enumerate(lines):
        pos = (x, y + line_h * (i + 1))
        cv2.putText(frame, line, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

# Sub-pixel corner refinement. Without this the corners land on whole pixels
# and the reprojection error never drops below about one pixel.
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# --------------------------------------------------------------------------
# Step 1 — capture
# --------------------------------------------------------------------------
def capture(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = (args.cols, args.rows)

    cap = open_source(int(args.source) if str(args.source).isdigit() else args.source)
    kept = 0

    print("Move the board between shots: tilt it, turn it, fill different parts")
    print("of the frame, vary the distance. Fifteen varied views beat fifty")
    print("identical ones — the solver needs disagreement to work with.\n")

    try:
        for frame in frames(cap):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
            )

            shown = frame.copy()
            if found:
                cv2.drawChessboardCorners(shown, pattern, corners, found)

            banner(shown, [
                f"board {'found' if found else 'not found'}    kept {kept}",
                "space keep    q done",
            ])
            cv2.imshow("Camera calibration: capture", shown)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if not found:
                    print("  no board in this frame — not saved")
                    continue
                kept += 1
                path = out_dir / f"shot_{kept:02d}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"  kept {path.name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n{kept} images in {out_dir}")
    if kept < 10:
        print("That is probably too few. Aim for fifteen to twenty.")


# --------------------------------------------------------------------------
# Step 2 — solve
# --------------------------------------------------------------------------
def solve(args):
    img_dir = Path(args.images)
    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise SystemExit(f"No images in {img_dir}")

    pattern = (args.cols, args.rows)

    # One template of the board's corners in its own coordinate frame, in mm.
    # z is zero for every corner because the board is flat — that flatness is
    # exactly what makes Zhang's method work from a printed sheet.
    template = np.zeros((args.rows * args.cols, 3), np.float32)
    template[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    template *= args.square

    object_points, image_points, used = [], [], []
    size = None

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  {path.name}: could not read")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"  {path.name}: board not found — skipped")
            continue

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        object_points.append(template)
        image_points.append(refined)
        used.append(path.name)
        print(f"  {path.name}: ok")

    if len(object_points) < 5:
        raise SystemExit(
            f"\nOnly {len(object_points)} usable views. Check that --cols and --rows "
            "are the INNER corner counts, not the number of squares."
        )

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, size, None, None
    )

    # Per-view reprojection error, so a bad photo can be identified and removed
    # rather than quietly dragging the whole solution.
    per_view = []
    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], K, dist)
        # Mean distance, in pixels, between where each corner was found and
        # where the solved camera says it should be.
        observed = image_points[i].reshape(-1, 2).astype(np.float64)
        expected = projected.reshape(-1, 2).astype(np.float64)
        err = float(np.linalg.norm(observed - expected, axis=1).mean())
        per_view.append((used[i], round(err, 4)))

    print("\n" + "=" * 56)
    print(f"views used          {len(object_points)} of {len(paths)}")
    print(f"image size          {size[0]} x {size[1]}")
    print(f"\nfx                  {K[0, 0]:10.2f} px")
    print(f"fy                  {K[1, 1]:10.2f} px")
    print(f"cx                  {K[0, 2]:10.2f} px   (image centre x = {size[0] / 2:.1f})")
    print(f"cy                  {K[1, 2]:10.2f} px   (image centre y = {size[1] / 2:.1f})")
    print(f"\ndistortion          k1={dist[0][0]:+.4f}  k2={dist[0][1]:+.4f}  "
          f"p1={dist[0][2]:+.4f}  p2={dist[0][3]:+.4f}  k3={dist[0][4]:+.4f}")
    print(f"\nreprojection error  {rms:.4f} px")
    print("                    below 1.0 is good, below 0.5 is very good")

    worst = max(per_view, key=lambda t: t[1])
    print(f"\nworst view          {worst[0]} at {worst[1]} px")
    print("                    remove it and solve again — does the error drop?")

    out = img_dir / "camera.json"
    out.write_text(json.dumps({
        "image_width": int(size[0]), "image_height": int(size[1]),
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "distortion": [float(v) for v in dist.ravel()],
        "reprojection_error_px": float(rms),
        "views_used": used,
        "per_view_error_px": dict(per_view),
        "square_size_mm": args.square,
    }, indent=2))
    print(f"\nwritten -> {out}")


# --------------------------------------------------------------------------
# Step 3 — measure
# --------------------------------------------------------------------------
def measure(args):
    calib = json.loads(Path(args.calib).read_text())
    fy = calib["fy"]

    if args.pixels and args.height_mm and not args.distance_mm:
        distance = fy * args.height_mm / args.pixels
        print(f"\nfy                {fy:.2f} px")
        print(f"object height     {args.height_mm:.1f} mm")
        print(f"height in image   {args.pixels:.1f} px")
        print(f"\ndistance          {distance:.1f} mm   ({distance / 1000:.2f} m)")
    elif args.pixels and args.distance_mm and not args.height_mm:
        height = args.pixels * args.distance_mm / fy
        print(f"\nobject height     {height:.1f} mm")
    else:
        raise SystemExit(
            "Give --pixels plus exactly one of --height-mm or --distance-mm.\n"
            "The equation has three unknowns; you supply two and it returns the third."
        )

    print("\nThis assumes the object is flat-on to the camera and near the centre")
    print("of the frame. Both assumptions are things you are asked to break in")
    print("step 4 — measure the error each time, do not just note that it got worse.")


# --------------------------------------------------------------------------
# Step 4 — undistort
# --------------------------------------------------------------------------
def undistort(args):
    calib = json.loads(Path(args.calib).read_text())
    K = np.array([[calib["fx"], 0, calib["cx"]],
                  [0, calib["fy"], calib["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(calib["distortion"], dtype=np.float64)

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"Could not read {args.image}")

    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    fixed = cv2.undistort(img, K, dist, None, newK)

    # Difference map: where the lens was actually moving pixels around.
    diff = cv2.absdiff(img, fixed)
    heat = cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)

    scale = 380 / h
    tiles = []
    for name, tile in [("original", img), ("undistorted", fixed), ("what moved", heat)]:
        small = cv2.resize(tile, (int(w * scale), 380))
        cv2.rectangle(small, (0, 0), (small.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(small, name, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(small)
    gallery = np.hstack(tiles)

    path = output_path("camera_calibration_undistort.jpg")
    cv2.imwrite(str(path), gallery)
    print(f"written -> {path}")
    print("\nLook at the third panel. The correction is near zero in the middle")
    print("and largest at the corners — which is why a measurement taken at the")
    print("edge of an uncalibrated frame is the one that fools you.")

    cv2.imshow("Lab 02c — distortion removed", gallery)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Calibrate your own camera and measure with it")
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("capture", help="collect checkerboard images")
    c.add_argument("--out", default="myboard")
    c.add_argument("--source", default=0)
    c.add_argument("--cols", type=int, default=9, help="inner corners across")
    c.add_argument("--rows", type=int, default=6, help="inner corners down")
    c.set_defaults(func=capture)

    s = sub.add_parser("solve", help="compute intrinsics and distortion")
    s.add_argument("--images", default="myboard")
    s.add_argument("--cols", type=int, default=9)
    s.add_argument("--rows", type=int, default=6)
    s.add_argument("--square", type=float, required=True, help="square size in mm")
    s.set_defaults(func=solve)

    m = sub.add_parser("measure", help="use the result to measure something")
    m.add_argument("--calib", required=True)
    m.add_argument("--pixels", type=float, help="object height in the image, in pixels")
    m.add_argument("--height-mm", type=float, help="real object height in mm")
    m.add_argument("--distance-mm", type=float, help="real distance in mm")
    m.set_defaults(func=measure)

    u = sub.add_parser("undistort", help="show what the distortion was doing")
    u.add_argument("--calib", required=True)
    u.add_argument("--image", required=True)
    u.set_defaults(func=undistort)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
