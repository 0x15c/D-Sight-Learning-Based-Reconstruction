import os
import json
import random
import cv2
from ultralytics import YOLO

# ---- config ----
MODEL_PATH = "./circle_detection_dataset/runs/detect/train4/weights/best.pt"
INPUT_VIDEO = "video/calib2.mp4"      # or 0 for webcam
OUT_DIR = "export_frames"
OUT_JSON = os.path.join(OUT_DIR, "circles.json")

CONF_THRES = 0.25
IOU_THRES = 0.5

MAX_EXPORT = 200
RANDOM_SEED = 0  # change for different random selection, or set None for non-deterministic
# ---------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ls_radius_from_box(w: float, h: float) -> float:
    # w ≈ 2r, h ≈ 2r  => LS r = (w+h)/4
    return (w + h) / 4.0

def box_to_circle_ls(x1: float, y1: float, x2: float, y2: float):
    w = x2 - x1
    h = y2 - y1
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    r = ls_radius_from_box(w, h)
    return cx, cy, r

def circle_fully_inside(cx: float, cy: float, r: float, W: int, H: int) -> bool:
    # touching/intersecting boundary -> reject
    return (cx - r >= 0) and (cy - r >= 0) and (cx + r < W) and (cy + r < H)

def main():
    ensure_dir(OUT_DIR)

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # -------- Pass 1: collect all valid frames into memory --------
    valid_items = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
        r0 = results[0]

        circles_this_frame = []

        if r0.boxes is not None and len(r0.boxes) > 0:
            boxes = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            clss  = r0.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

                cx, cy, rad = box_to_circle_ls(x1, y1, x2, y2)

                if rad <= 0:
                    continue
                if not circle_fully_inside(cx, cy, rad, W, H):
                    continue

                circles_this_frame.append({
                    "center_px": [float(cx), float(cy)],
                    "radius_px": float(rad),
                    "conf": float(conf),
                    "class_id": int(cls_id),
                    "class_name": model.names.get(int(cls_id), str(cls_id)),
                })

        # Valid frame = at least 1 valid circle
        if len(circles_this_frame) > 0:
            valid_items.append({
                "frame_idx": frame_idx,
                "frame": frame,  # raw, unannotated
                "circles": circles_this_frame
            })

        frame_idx += 1

    cap.release()

    total_valid = len(valid_items)
    if total_valid == 0:
        # Still write a JSON with info so downstream code doesn't explode
        export = {"info": {"width": W, "height": H}, "frames": {}}
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2)
        print("No valid frames found. Wrote empty JSON.")
        return

    # -------- Pass 2: randomly choose up to MAX_EXPORT and export --------
    k = min(MAX_EXPORT, total_valid)
    chosen = random.sample(valid_items, k=k)

    export = {
        "info": {"width": W, "height": H},
        "frames": {}
    }

    # Optional: make filenames stable & meaningful
    # (keep original frame index in the filename)
    for item in chosen:
        idx = item["frame_idx"]
        fname = f"frame_{idx:06d}.png"
        fpath = os.path.join(OUT_DIR, fname)

        ok = cv2.imwrite(fpath, item["frame"])
        if not ok:
            raise RuntimeError(f"Failed to write image: {fpath}")

        export["frames"][fname] = item["circles"]

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

    print("Done.")
    print(f"Video size: {W}x{H}")
    print(f"Total frames processed: {frame_idx}")
    print(f"Valid frames found: {total_valid}")
    print(f"Frames exported (random): {k}")
    print(f"Images folder: {OUT_DIR}")
    print(f"JSON path: {OUT_JSON}")

if __name__ == "__main__":
    main()
