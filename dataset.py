"""Cropped calibration images and spherical-cap heights, cached once for both splits."""

import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class HeightDataset(Dataset):
    def __init__(self, config):
        self.root = config.resolve(config.data["paths"]["dataset"])
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.samples = self.manifest["samples"]
        # null keeps the complete cropped camera grid. Explicit sizes remain
        # supported for datasets/checkpoints made with the earlier workflow.
        configured_size = config.data["training"].get("image_size")
        self.image_size = tuple(self.manifest["image_size"] if configured_size is None else configured_size)
        height, width = self.image_size
        count = len(self.samples)

        # RGB stays uint8 until it reaches the GPU. Heights are float32 millimetres.
        # Reserve room for decoded source frames and queued/pinned training batches.
        train = config.data["training"]
        self.cache_bytes = count * height * width * 7
        source_h, source_w = self.manifest["image_size"]
        reserve_bytes = 4 * train["batch_size"] * height * width * 7 + source_h * source_w * 24
        budget = min(float(train["cache_limit_gib"]), 32.0) * 1024**3
        self.cached = self.cache_bytes + reserve_bytes <= budget
        self.images = self.heights = None
        if self.cached:
            # Allocate directly; avoid list -> stack and separate train/val copies.
            self.images = torch.empty((count, 3, height, width), dtype=torch.uint8)
            self.heights = torch.empty((count, height, width), dtype=torch.float32)
            for index in range(count):
                image, depth = self.read_sample(index)
                self.images[index].copy_(image)
                self.heights[index].copy_(depth)
            print(f"Cached {count} samples: {self.cache_bytes / 1024**3:.2f} GiB.")
        else:
            print(f"Streaming {count} samples; cache would need {self.cache_bytes / 1024**3:.2f} GiB.")

    def __len__(self):
        return len(self.samples)

    def read_sample(self, index):
        sample = self.samples[index]
        height, width = self.image_size
        image = cv2.imread(str(self.root / sample["image"]), cv2.IMREAD_COLOR)
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = np.load(self.root / sample["height"])[..., 0]
        if depth.shape != self.image_size:
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_AREA)
        # Resizing changes sampling density, never the physical height in mm.
        return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(depth)

    def __getitem__(self, index):
        if self.cached:
            return self.images[index], self.heights[index]
        return self.read_sample(index)


def load_background(config, image_size):
    """The camera setup stores an unloaded, cropped BGR image."""
    if not config.data["training"]["subtract_background"]:
        return None
    path = config.resolve(config.data["paths"]["background"])
    background = cv2.imread(str(path), cv2.IMREAD_COLOR)
    height, width = image_size
    if background.shape[:2] != tuple(image_size):
        background = cv2.resize(background, (width, height), interpolation=cv2.INTER_AREA)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(background).permute(2, 0, 1).unsqueeze(0).float() / 255.0
