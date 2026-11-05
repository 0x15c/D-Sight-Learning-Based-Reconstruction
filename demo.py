"""Select a sensor and display live RGB, depth (mm), and optional point cloud."""
import copy
import time

import cv2
import numpy as np
import torch

from camera import Camera
from model import UNet2D
from poisson_solver import PoissonSolver
from sensor_config import SensorConfig

# Set True to always show the cloud, False to hide it, or None to use the config.
SHOW_POINT_CLOUD = True


class Reconstructor:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = config.data["paths"]["checkpoint"]
        if not checkpoint_path:
            raise RuntimeError("Train this sensor with train.py before opening the demo.")
        self.checkpoint = torch.load(config.resolve(checkpoint_path), map_location="cpu", weights_only=True)
        self.height, self.width = self.checkpoint["image_size"]
        self.model = UNet2D(base_channels=self.checkpoint["base_channels"]).to(self.device).eval()
        self.model.load_state_dict(self.checkpoint["model_state"])
        # The gradient model was optimized through this integration convention.
        # Older checkpoints were trained with a fixed-zero rim.
        self.reconstruction = self.checkpoint.get("reconstruction", {
            "boundary": "dirichlet", "reference_region": None})
        self.solver = PoissonSolver(
            self.height, self.width, boundary=self.reconstruction["boundary"],
            reference_region=self.reconstruction.get("reference_region"),
        ).to(self.device)
        self.background = self.checkpoint["background"]
        if self.background is not None:
            self.background = self.background.to(self.device)
        self.amp = self.device.type == "cuda" and config.data["training"]["amp"]
        self.max_depth = config.data["demo"]["max_depth_mm"]
        self.min_depth = config.data["demo"].get("min_depth_mm")
        if self.min_depth is None:
            self.min_depth = -self.max_depth if self.reconstruction["boundary"] == "neumann" else 0.0
        lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_TURBO)
        self.colors = torch.from_numpy(lut[:, 0].copy()).to(self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        print(f"Reconstruction: {self.device}, {self.width} x {self.height}, depth in mm, "
              f"{self.reconstruction['boundary']} boundary")
        requested_boundary = config.data.get("reconstruction", {}).get("boundary", "dirichlet")
        if requested_boundary != self.reconstruction["boundary"]:
            print(f"This checkpoint was trained with {self.reconstruction['boundary']} boundaries. "
                  f"Run train.py to train with the configured {requested_boundary} solver; "
                  "the existing checkpoint keeps its original solver.")
        source_height, source_width = self.checkpoint["source_image_size"]
        if config.data["training"].get("image_size") is None and (self.height, self.width) != (source_height, source_width):
            print(f"This checkpoint uses the earlier resized input. Run train.py to train at "
                  f"the native {source_width} x {source_height} crop resolution.")

    @torch.inference_mode()
    def __call__(self, cropped_bgr):
        # Native checkpoints use the exact crop grid, including odd dimensions.
        # Older resized checkpoints retain their original preprocessing.
        if cropped_bgr.shape[:2] == (self.height, self.width):
            bgr = cropped_bgr
        else:
            bgr = cv2.resize(cropped_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).float().div_(255)
        if self.background is not None:
            tensor = tensor - self.background
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            gradients = self.model(tensor)
        depth = self.solver(gradients)[0]
        indices = ((depth - self.min_depth) / (self.max_depth - self.min_depth) * 255).clamp(0, 255).long()
        heat = self.colors[indices].cpu().numpy()
        return bgr, heat, depth


class DepthCloud:
    def __init__(self, reconstructor, settings):
        import pyvista as pv

        self.stride = settings["point_stride"]
        self.update_interval = 1.0 / settings.get("cloud_fps", 15)
        self.next_update = 0.0
        h, w = reconstructor.height, reconstructor.width
        src_h, src_w = reconstructor.checkpoint["source_image_size"]
        scale = reconstructor.checkpoint["mm_per_pixel"]
        y, x = torch.meshgrid(
            torch.arange(0, h, self.stride, device=reconstructor.device),
            torch.arange(0, w, self.stride, device=reconstructor.device), indexing="ij")
        self.points = torch.stack((x * (src_w / w) * scale,
                                   y * (src_h / h) * scale, torch.zeros_like(x)), dim=-1).float()
        self.cloud = pv.PolyData(self.points.reshape(-1, 3).cpu().numpy())
        self.cloud["Depth (mm)"] = np.zeros(self.cloud.n_points, dtype=np.float32)
        self.plotter = pv.Plotter(title="D-Sight depth (mm)")
        self.plotter.add_mesh(self.cloud, scalars="Depth (mm)", style="points", point_size=3,
                              render_points_as_spheres=False, cmap="turbo",
                              clim=(reconstructor.min_depth, reconstructor.max_depth))
        self.plotter.show_axes()
        self.plotter.show(interactive_update=True, auto_close=False)

    def update(self, depth):
        # 3D rendering has its own cadence so it cannot force a redraw per frame.
        if time.perf_counter() < self.next_update:
            return
        z = depth[::self.stride, ::self.stride].reshape(-1).cpu().numpy()
        # X/Y are fixed. Update existing VTK arrays instead of replacing the cloud.
        self.cloud.points[:, 2] = z
        self.cloud["Depth (mm)"][:] = z
        self.plotter.update()
        self.next_update = time.perf_counter() + self.update_interval

    def close(self):
        self.plotter.close()


def main(video_path=None):
    config = SensorConfig.select()
    reconstructor = Reconstructor(config)
    # Replay the training camera settings/crop; the device index may have changed.
    capture_config = copy.copy(config)
    capture_config.data = copy.deepcopy(config.data)
    capture_config.data["camera"] = copy.deepcopy(reconstructor.checkpoint["camera"])
    capture_config.data["camera"]["index"] = config.data["camera"]["index"]
    capture_config.data["camera"]["fourcc"] = config.data["camera"].get("fourcc", "MJPG")
    camera = None
    video = None
    cloud = None
    show_cloud = config.data["demo"]["pyvista"] if SHOW_POINT_CLOUD is None else SHOW_POINT_CLOUD
    window = "D-Sight | RGB and depth | Q to close"
    try:
        if video_path is None:
            camera = Camera(capture_config)
        else:
            video = cv2.VideoCapture(str(config.resolve(video_path)))
            if not video.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
        if show_cloud:
            cloud = DepthCloud(reconstructor, config.data["demo"])
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        previous = time.perf_counter()
        fps = None
        while True:
            if camera is not None:
                frame = camera.read()
                cropped = camera.crop(frame)
            else:
                ok, frame = video.read()
                if not ok:
                    break
                crop = capture_config.data["camera"]["crop"]
                if crop is None:
                    cropped = frame
                else:
                    x, y, w, h = crop
                    cropped = frame[y:y + h, x:x + w]
            bgr, heat, depth = reconstructor(cropped)
            if cloud is not None:
                cloud.update(depth)
            now = time.perf_counter()
            current_fps = 1.0 / (now - previous)
            fps = current_fps if fps is None else 0.9 * fps + 0.1 * current_fps
            previous = now
            rate_text = f"Display {fps:.1f} FPS"
            if camera is not None:
                rate_text += f" | Camera {camera.fps:.1f}"
            # cv2.putText(heat, rate_text, (8, 22),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(heat, f"Depth: {reconstructor.min_depth:g} to {reconstructor.max_depth:g} mm", (8, 42),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, np.concatenate((bgr, heat), axis=1))
            if cv2.pollKey() & 0xFF in (27, ord("q")):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if camera is not None:
            camera.close()
        if video is not None:
            video.release()
        if cloud is not None:
            cloud.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
