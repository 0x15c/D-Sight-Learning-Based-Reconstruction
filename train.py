"""Select a sensor, then train its U-Net with reconstructed-height supervision."""

import copy
import csv
import math
import time

import torch
from torch.utils.data import DataLoader, Subset

from dataset import HeightDataset, load_background
from loss import ReconstructionLoss
from model import UNet2D
from sensor_config import SensorConfig


class Trainer:
    def __init__(self, config):
        self.config = config
        settings = config.data["training"]
        torch.manual_seed(settings["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = bool(settings["amp"] and self.device.type == "cuda")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        else:
            print("CUDA is unavailable; training on CPU.")

        self.dataset = HeightDataset(config)
        if len(self.dataset) < 2:
            raise RuntimeError("Build at least two accepted calibration samples before training.")
        generator = torch.Generator().manual_seed(settings["seed"])
        order = torch.randperm(len(self.dataset), generator=generator).tolist()
        n_val = max(1, min(len(order) - 1, round(len(order) * settings["validation_fraction"])))
        val_indices, train_indices = order[:n_val], order[n_val:]

        # Shared Subsets refer to the same cache. Workers only help disk streaming;
        # using them with a cached dataset would duplicate it under Windows spawn.
        workers = 0 if self.dataset.cached else settings["num_workers"]
        loader_options = dict(
            batch_size=settings["batch_size"],
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
        )
        if workers:
            loader_options.update(persistent_workers=True, prefetch_factor=2)
        self.train_loader = DataLoader(
            Subset(self.dataset, train_indices), shuffle=True, generator=generator, **loader_options
        )
        self.val_loader = DataLoader(Subset(self.dataset, val_indices), **loader_options)
        self.background = load_background(config, self.dataset.image_size)
        if self.background is not None:
            self.background = self.background.to(self.device)

        self.net = UNet2D(in_channels=3, base_channels=settings["base_channels"]).to(self.device)
        reconstruction = config.data.get("reconstruction", {})
        region = reconstruction.get("reference_region")
        if region is not None:
            # Config coordinates refer to the native crop; checkpoint coordinates
            # refer to model inputs so inference can replay the same reference.
            source_h, source_w = self.dataset.manifest["image_size"]
            input_h, input_w = self.dataset.image_size
            x, y, width, height = region
            x0 = min(input_w - 1, max(0, math.floor(x * input_w / source_w)))
            y0 = min(input_h - 1, max(0, math.floor(y * input_h / source_h)))
            x1 = min(input_w, max(x0 + 1, math.ceil((x + width) * input_w / source_w)))
            y1 = min(input_h, max(y0 + 1, math.ceil((y + height) * input_h / source_h)))
            region = [x0, y0, x1 - x0, y1 - y0]
        self.reconstruction = {
            "boundary": reconstruction.get("boundary", "dirichlet"),
            "reference_region": region,
        }
        if self.reconstruction["boundary"] == "neumann":
            # A random constant gradient becomes a large tilted plane under free
            # boundaries. Start a newly trained model from a flat unloaded gel.
            torch.nn.init.zeros_(self.net.grad_head.weight)
            torch.nn.init.zeros_(self.net.grad_head.bias)
        self.loss = ReconstructionLoss(
            self.dataset.image_size, settings["gradient_loss_weight"], **self.reconstruction
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=settings["learning_rate"])
        # Free-boundary integration amplifies low-frequency gradient errors at
        # native resolution; a smaller initial scale avoids startup FP16 overflow.
        initial_scale = 64.0 if self.reconstruction["boundary"] == "neumann" else 65536.0
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp, init_scale=initial_scale)
        self.output = self.dataset.root.parent / "checkpoints"
        self.output.mkdir(parents=True, exist_ok=True)
        height, width = self.dataset.image_size
        print(f"{self.device}: {len(train_indices)} train / {len(val_indices)} validation samples, "
              f"U-Net input {width} x {height} pixels, {self.reconstruction['boundary']} boundary.")

    def run_epoch(self, training):
        self.net.train(training)
        loader = self.train_loader if training else self.val_loader
        totals = torch.zeros(3, device=self.device)
        count = 0
        with torch.set_grad_enabled(training):
            for images, heights in loader:
                # DataLoader pins uint8 images; transfer fewer bytes, normalize here.
                images = images.to(self.device, non_blocking=True).float().div_(255.0)
                heights = heights.to(self.device, non_blocking=True)
                if self.background is not None:
                    images.sub_(self.background)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                    gradients = self.net(images)
                total, depth_loss, gradient_loss = self.loss(gradients, heights)
                if training:
                    self.scaler.scale(total).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                batch_size = images.shape[0]
                totals += torch.stack((total.detach(), depth_loss.detach(), gradient_loss.detach())) * batch_size
                count += batch_size
        # One GPU synchronization per epoch, rather than .item() for every batch.
        return (totals / count).cpu().tolist()

    def run(self):
        settings = self.config.data["training"]
        best = float("inf")
        suffix = "_neumann" if self.reconstruction["boundary"] == "neumann" else ""
        checkpoint_path = self.output / f"best{suffix}.pt"
        with (self.output / f"history{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
            history = csv.writer(handle)
            history.writerow(("epoch", "train_loss", "train_depth_mse", "train_gradient_mse",
                              "val_loss", "val_depth_mse", "val_gradient_mse", "seconds"))
            for epoch in range(1, settings["epochs"] + 1):
                start = time.perf_counter()
                train_metrics = self.run_epoch(training=True)
                val_metrics = self.run_epoch(training=False)
                elapsed = time.perf_counter() - start
                history.writerow((epoch, *train_metrics, *val_metrics, elapsed))
                handle.flush()
                print(f"Epoch {epoch:03d}/{settings['epochs']} | train {train_metrics[0]:.6g} "
                      f"| val {val_metrics[0]:.6g} | depth MSE {val_metrics[1]:.6g} mm^2 "
                      f"| {elapsed:.1f}s")
                if val_metrics[0] < best:
                    best = val_metrics[0]
                    manifest = self.dataset.manifest
                    torch.save({
                        "format_version": 2,
                        "model_state": self.net.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "train_loss": train_metrics[0],
                        "val_loss": val_metrics[0],
                        "base_channels": settings["base_channels"],
                        "image_size": list(self.dataset.image_size),
                        "subtract_background": settings["subtract_background"],
                        "background": self.background.cpu() if self.background is not None else None,
                        "mm_per_pixel": manifest["mm_per_pixel"],
                        "source_image_size": manifest["image_size"],
                        "ball_diameter_mm": manifest["ball_diameter_mm"],
                        "camera": copy.deepcopy(manifest.get("camera", self.config.data["camera"])),
                        "sensor": copy.deepcopy(self.config.data["sensor"]),
                        "training": copy.deepcopy(settings),
                        "reconstruction": copy.deepcopy(self.reconstruction),
                    }, checkpoint_path)
                    self.config.data["paths"]["checkpoint"] = checkpoint_path.as_posix()
                    self.config.save()
        print(f"Best validation loss: {best:.6g}. Checkpoint: {checkpoint_path}")
        return checkpoint_path


def main():
    config = SensorConfig.select()
    Trainer(config).run()


if __name__ == "__main__":
    main()
