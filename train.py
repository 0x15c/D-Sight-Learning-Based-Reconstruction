import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import wandb

import model


# ---- Tunable config ----
IMAGE_DIR = "export_frames"
JSON_PATH = os.path.join("export_frames", "balls.json")
BALL_RADIUS_MM = 7/2
MM_PER_PX = 34/1460
VAL_SPLIT = 0.2
SEED = 42
EPOCHS = 100
BATCH_SIZE = 4
LR = 1e-3
BASE_CHANNELS = 16
MAX_SAMPLES = None
RESIZE_H = None
RESIZE_W = None
WANDB_PROJECT = "d-sight-reconstruction"
WANDB_RUN_NAME = None
PREVIEW = True
PREVIEW_EVERY_EPOCH = 1
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY = 10
# ------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_annotations(json_path: str) -> Dict[str, List[dict]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("frames", {})


def select_best_circle(circles: List[dict]) -> dict:
    if len(circles) == 1:
        return circles[0]
    return max(circles, key=lambda c: c.get("conf", 0.0))


def build_height_map(
    circle: dict,
    height: int,
    width: int,
    ball_radius_mm: float,
    mm_per_px: float,
) -> torch.Tensor:
    cx, cy = circle["center_px"]
    r_px = circle["radius_px"]

    r_mm = float(r_px) * mm_per_px
    R = float(ball_radius_mm)
    if r_mm >= R:
        r_mm = R * 0.999

    yy, xx = np.ogrid[:height, :width]
    dx = xx - float(cx)
    dy = yy - float(cy)
    r_px_grid = np.sqrt(dx * dx + dy * dy)
    r_mm_grid = r_px_grid * mm_per_px

    z = np.zeros((height, width), dtype=np.float32)
    mask = r_mm_grid <= r_mm
    base = np.sqrt(max(R * R - r_mm * r_mm, 1e-8))
    z[mask] = np.sqrt(R * R - r_mm_grid[mask] ** 2) - base
    return torch.from_numpy(z)


def load_dataset_to_memory(
    image_dir: str,
    json_path: str,
    ball_radius_mm: float,
    mm_per_px: float,
    val_split: float,
    seed: int,
    max_samples: Optional[int],
    resize_hw: Optional[Tuple[int, int]],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    frames = load_annotations(json_path)
    keys = sorted(frames.keys())

    if max_samples is not None:
        keys = keys[: max_samples]

    images: List[torch.Tensor] = []
    heights: List[torch.Tensor] = []

    for fname in keys:
        circles = frames[fname]
        if not circles:
            continue
        circle = select_best_circle(circles)
        fpath = os.path.join(image_dir, fname)
        img_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if resize_hw is not None:
            img_rgb = cv2.resize(img_rgb, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA)

        h, w, _ = img_rgb.shape
        gt = build_height_map(circle, h, w, ball_radius_mm, mm_per_px)

        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).contiguous()
        images.append(img_tensor)
        heights.append(gt)

    if not images:
        raise RuntimeError("No valid images loaded. Check your paths and JSON.")

    imgs = torch.stack(images, dim=0)
    gts = torch.stack(heights, dim=0)

    total = imgs.shape[0]
    indices = np.arange(total)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(total * (1.0 - val_split))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_imgs = imgs[train_idx]
    train_gts = gts[train_idx]
    val_imgs = imgs[val_idx]
    val_gts = gts[val_idx]

    return (train_imgs, train_gts), (val_imgs, val_gts)


class InMemoryDataset(Dataset):
    def __init__(self, images: torch.Tensor, heights: torch.Tensor):
        self.images = images
        self.heights = heights

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.heights[idx]


def integrate_depth_batch(grad_bchw: torch.Tensor) -> torch.Tensor:
    depths = []
    for i in range(grad_bchw.shape[0]):
        grad_hw2 = grad_bchw[i].permute(1, 2, 0).contiguous()
        depths.append(model.getDepth(grad_hw2))
    return torch.stack(depths, dim=0)


def zero_mean(z: torch.Tensor) -> torch.Tensor:
    return z - z.mean(dim=(1, 2), keepdim=True)


def height_to_grad(z: torch.Tensor) -> torch.Tensor:
    # z: (B, H, W) -> grad: (B, 2, H, W)
    dx = torch.zeros_like(z)
    dy = torch.zeros_like(z)
    dx[:, :, 1:-1] = 0.5 * (z[:, :, 2:] - z[:, :, :-2])
    dy[:, 1:-1, :] = 0.5 * (z[:, 2:, :] - z[:, :-2, :])
    dx[:, :, 0] = z[:, :, 1] - z[:, :, 0]
    dx[:, :, -1] = z[:, :, -1] - z[:, :, -2]
    dy[:, 0, :] = z[:, 1, :] - z[:, 0, :]
    dy[:, -1, :] = z[:, -1, :] - z[:, -2, :]
    return torch.stack([dx, dy], dim=1)


def _normalize_to_uint8(img: np.ndarray, symmetric: bool = False) -> np.ndarray:
    if symmetric:
        max_abs = float(np.max(np.abs(img))) if img.size else 0.0
        denom = max(max_abs, 1e-6)
        norm = (img / denom + 1.0) * 0.5
    else:
        min_v = float(np.min(img)) if img.size else 0.0
        max_v = float(np.max(img)) if img.size else 1.0
        denom = max(max_v - min_v, 1e-6)
        norm = (img - min_v) / denom
    out = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return out


def preview_validation(
    imgs: torch.Tensor,
    pred_grad: torch.Tensor,
    gt_height: torch.Tensor,
    # window_prefix: str = "val",
) -> None:
    # cv2.destroyAllWindows()
    # Show only the first sample in batch to keep UI responsive.
    img = imgs[0].detach().cpu().numpy()  # (3, H, W)
    grad = pred_grad[0].detach().cpu().numpy()  # (2, H, W)
    gt = gt_height[0].detach().cpu().numpy()  # (H, W)

    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    grad_x = grad[0]
    grad_y = grad[1]

    grad_vis = np.zeros((grad_x.shape[0], grad_x.shape[1], 3), dtype=np.uint8)
    grad_vis[:, :, 0] = _normalize_to_uint8(grad_x, symmetric=True)  # B
    grad_vis[:, :, 2] = _normalize_to_uint8(grad_y, symmetric=True)  # R

    gt_height_vis = _normalize_to_uint8(gt, symmetric=False)
    gt_height_vis = cv2.applyColorMap(gt_height_vis, cv2.COLORMAP_TURBO)

    gt_grad = height_to_grad(torch.from_numpy(gt[None, ...])).numpy()[0]
    gt_grad_x = gt_grad[0]
    gt_grad_y = gt_grad[1]
    gt_grad_vis = np.zeros((gt_grad_x.shape[0], gt_grad_x.shape[1], 3), dtype=np.uint8)
    gt_grad_vis[:, :, 0] = _normalize_to_uint8(gt_grad_x, symmetric=True)
    gt_grad_vis[:, :, 2] = _normalize_to_uint8(gt_grad_y, symmetric=True)

    cv2.imshow("input", img_bgr)
    cv2.imshow("pred_grad (B=x, R=y)", grad_vis)
    cv2.imshow("gt_height", gt_height_vis)
    cv2.imshow("gt_grad (B=x, R=y)", gt_grad_vis)
    cv2.waitKey(1)


def train_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    net.train()
    mse = nn.MSELoss()
    running = 0.0
    count = 0
    for imgs, gt in loader:
        imgs = imgs.to(device)
        gt = gt.to(device)

        optimizer.zero_grad(set_to_none=True)
        grad = net(imgs)
        depth = integrate_depth_batch(grad)
        loss = mse(zero_mean(depth), zero_mean(gt))
        loss.backward()
        optimizer.step()

        running += loss.item() * imgs.shape[0]
        count += imgs.shape[0]
    return running / max(count, 1)


@torch.no_grad()
def eval_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    net.eval()
    mse = nn.MSELoss()
    running = 0.0
    count = 0
    shown = False
    for imgs, gt in loader:
        imgs = imgs.to(device)
        gt = gt.to(device)
        grad = net(imgs)
        depth = integrate_depth_batch(grad)
        loss = mse(zero_mean(depth), zero_mean(gt))
        running += loss.item() * imgs.shape[0]
        count += imgs.shape[0]

        if PREVIEW and not shown and (epoch % PREVIEW_EVERY_EPOCH == 0):
            preview_validation(imgs, grad, gt)
            shown = True
    return running / max(count, 1)


def main() -> None:
    if BALL_RADIUS_MM is None or MM_PER_PX is None:
        raise ValueError("Set BALL_RADIUS_MM and MM_PER_PX in train.py before running.")

    set_seed(SEED)

    resize_hw = None
    if RESIZE_H is not None and RESIZE_W is not None:
        resize_hw = (RESIZE_H, RESIZE_W)

    (train_imgs, train_gts), (val_imgs, val_gts) = load_dataset_to_memory(
        image_dir=IMAGE_DIR,
        json_path=JSON_PATH,
        ball_radius_mm=BALL_RADIUS_MM,
        mm_per_px=MM_PER_PX,
        val_split=VAL_SPLIT,
        seed=SEED,
        max_samples=MAX_SAMPLES,
        resize_hw=resize_hw,
    )

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "image_dir": IMAGE_DIR,
            "json_path": JSON_PATH,
            "ball_radius_mm": BALL_RADIUS_MM,
            "mm_per_px": MM_PER_PX,
            "val_split": VAL_SPLIT,
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "base_channels": BASE_CHANNELS,
            "max_samples": MAX_SAMPLES,
            "resize_h": RESIZE_H,
            "resize_w": RESIZE_W,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.UNet2D(in_channels=3, base_channels=BASE_CHANNELS).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    train_ds = InMemoryDataset(train_imgs, train_gts)
    val_ds = InMemoryDataset(val_imgs, val_gts)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(net, train_loader, optimizer, device)
        val_loss = eval_one_epoch(net, val_loader, device, epoch)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        if CHECKPOINT_EVERY and (epoch % CHECKPOINT_EVERY == 0):
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": net.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()
