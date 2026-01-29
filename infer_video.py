import os
import time

import cv2
import numpy as np
import torch
from typing import Optional, Tuple

import model


# ---- Tunable config ----
VIDEO_PATH = "video/eval2_cropped.mp4"
CHECKPOINT_PATH = os.path.join("checkpoints", "epoch_100.pt")
OUTPUT_PATH = "video/pred_heatmap.mp4"
BASE_CHANNELS = 16
RESIZE_H = None
RESIZE_W = None
DISPLAY_SCALE = 1.0
WINDOW_NAME = "pred_height"
BACKGROUND_PATH = "export_frames/background.png"
# ------------------------


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    min_v = float(np.min(img)) if img.size else 0.0
    max_v = float(np.max(img)) if img.size else 1.0
    denom = max(max_v - min_v, 1e-6)
    norm = (img - min_v) / denom
    out = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return out


def load_model(device: torch.device) -> torch.nn.Module:
    net = model.UNet2D(in_channels=3, base_channels=BASE_CHANNELS).to(device)
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        state = ckpt.get("model_state", ckpt)
        net.load_state_dict(state)
    net.eval()
    return net


def load_background(path: Optional[str], resize_hw: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
    if not path:
        return None
    bg_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bg_bgr is None:
        raise RuntimeError(f"Could not read background image: {path}")
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    if resize_hw is not None:
        bg_rgb = cv2.resize(bg_rgb, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA)
    return bg_rgb.astype(np.float32) / 255.0


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(device)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    resize_hw = None
    if RESIZE_H is not None and RESIZE_W is not None:
        resize_hw = (RESIZE_H, RESIZE_W)
    background_rgb = load_background(BACKGROUND_PATH, resize_hw)

    writer = None
    prev_time = time.perf_counter()
    fps = 0.0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        orig_bgr = frame_bgr
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if RESIZE_H is not None and RESIZE_W is not None:
            frame_rgb = cv2.resize(frame_rgb, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0
        if background_rgb is not None:
            frame_rgb = frame_rgb - background_rgb

        img = torch.from_numpy(frame_rgb).float()
        img = img.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)

        with torch.no_grad():
            grad = net(img)
            grad_hw2 = grad[0].permute(1, 2, 0).contiguous()
            pred_z = model.getDepth(grad_hw2)

        pred_z_np = pred_z.detach().cpu().numpy()
        heat = _normalize_to_uint8(pred_z_np)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)

        if RESIZE_H is not None and RESIZE_W is not None:
            orig_bgr = cv2.resize(orig_bgr, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
        heat_bgr = heat

        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(
            heat_bgr,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        concat = np.concatenate([orig_bgr, heat_bgr], axis=1)

        if writer is None:
            h, w = concat.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                OUTPUT_PATH, fourcc, cap.get(cv2.CAP_PROP_FPS) or 30.0, (w, h)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open writer: {OUTPUT_PATH}")

        writer.write(concat)

        display = concat
        if DISPLAY_SCALE != 1.0:
            h, w = display.shape[:2]
            display = cv2.resize(
                display, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)), interpolation=cv2.INTER_AREA
            )

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
