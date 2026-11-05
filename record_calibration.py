"""Select a sensor, then collect annotation or reconstruction-training frames."""

import time
import copy
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import cv2

from camera import Camera
from ball_detector import BallDetector
from sensor_config import SensorConfig


class FrameWriter:
    """Encode full-resolution PNGs off the UI thread, with a small bounded queue."""
    def __init__(self, capacity=4, manifest_path=None, metadata=None):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="D-Sight PNG")
        self.pending = deque()
        self.capacity = capacity
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.manifest = None
        if self.manifest_path is not None:
            self.manifest = copy.deepcopy(metadata)
            self.manifest["samples"] = []
            if self.manifest_path.exists():
                previous = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                for key in ("ball_diameter_mm", "mm_per_pixel", "camera"):
                    if previous[key] != self.manifest[key]:
                        raise RuntimeError("Recording settings changed. Use a new paths.recording_images folder for this acquisition.")
                self.manifest = previous

    def poll(self):
        while self.pending and self.pending[0].done():
            if not self.pending.popleft().result():
                raise RuntimeError("Could not save calibration image.")

    def _write(self, path, frame, contact):
        if self.manifest is not None:
            size = list(frame.shape[:2])
            if self.manifest.get("image_size", size) != size:
                raise RuntimeError("Recording crop size changed; start a new acquisition folder.")
        saved = cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if saved and self.manifest is not None:
            self.manifest["image_size"] = size
            self.manifest["samples"].append({"image": Path(path).name, **contact})
            temporary = self.manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
            temporary.replace(self.manifest_path)
        return saved

    def submit(self, path, frame, contact=None):
        self.poll()
        if len(self.pending) >= self.capacity:
            return False
        self.pending.append(self.pool.submit(self._write, path, frame.copy(), copy.deepcopy(contact)))
        return True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.pool.shutdown(wait=True)
        self.poll()


class ContactCapture:
    """Admit each validated inference frame once, with a UI hold through release."""
    def __init__(self, calibration):
        self.interval = calibration["capture_interval_s"]
        self.release_delay = calibration.get("release_delay_s", 0.6)
        self.reset()

    def reset(self, now=float("-inf")):
        self.armed_since = now
        self.last_seen = self.last_saved = float("-inf")
        self.last_detection = None
        self.active = False
        self.status = "Waiting for detector"

    def update(self, detection, now):
        valid = False
        if detection is not None and detection["timestamp"] >= self.armed_since:
            if self.last_detection is None or detection["timestamp"] > self.last_detection["timestamp"]:
                self.last_detection = detection
                valid = detection["contact"] is not None
                if valid:
                    self.last_seen = detection["timestamp"]
        # The release hold affects UI state only. Every saved frame needs its own
        # valid detection, including while the ball is leaving the surface.
        self.active = now - self.last_seen <= self.release_delay
        if self.active:
            self.status = "Valid contact" if self.last_detection["contact"] is not None else "Waiting for valid release contact"
        else:
            self.status = "Waiting for ball" if self.last_detection is not None else "Waiting for detector"
        if self.last_detection is not None and self.last_detection["detected"] and self.last_detection["contact"] is None:
            self.status = "Rejected: contact geometry"
        return valid and detection["timestamp"] - self.last_saved >= self.interval


def choose_mode():
    window = tk.Tk()
    window.title("Calibration capture")
    result = []
    ttk.Label(window, text="Choose the images to record", padding=15).pack()
    for mode, label in (("annotation", "About 100 images for manual annotation"),
                        ("recording", "About 2000 images - automatic ball-triggered capture")):
        def choose(value=mode):
            result.append(value)
            window.destroy()
        ttk.Button(window, text=label, command=choose, padding=12).pack(fill="x", padx=15, pady=8)
    window.mainloop()
    if not result:
        raise SystemExit
    return result[0]


def main():
    config = SensorConfig.select()
    mode = choose_mode()
    output = config.resolve(config.data["paths"]["annotation_images" if mode == "annotation" else "recording_images"])
    output.mkdir(parents=True, exist_ok=True)
    calibration = config.data["calibration"]
    recording = mode == "recording"
    contact = ContactCapture(calibration) if recording else None
    # Check the trained checkpoint before opening the camera or display window.
    detector = BallDetector(config) if recording else None
    target = calibration["annotation_count" if mode == "annotation" else "record_count"]
    interval = calibration["capture_interval_s"]
    number = max((int(p.stem.split("_")[-1]) for p in output.glob("frame_*.png")), default=0)
    captured, automatic, last_capture = 0, False, 0.0
    paused = False
    candidate = None
    rejected = 0
    metadata = {"ball_diameter_mm": calibration["ball_diameter_mm"],
                "mm_per_pixel": calibration["mm_per_pixel"],
                "camera": copy.deepcopy(config.data["camera"]), "units": "mm"} if recording else None
    title = f"{config.data['sensor']['name']} - {mode}"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cropped, busy = None, False
    display_fps, displayed, fps_start = 0.0, 0, time.perf_counter()
    try:
        with (detector if detector is not None else nullcontext()) as detector:
            with Camera(config) as camera, FrameWriter(
                    manifest_path=output / "contacts.json" if recording else None, metadata=metadata) as writer:
                while captured < target:
                    writer.poll()
                    frame = camera.read(timeout=0)
                    now = time.monotonic()
                    fresh_crop = camera.crop(frame) if frame is not None else None
                    if detector is not None:
                        detection = detector.update(fresh_crop, now)
                        due = contact.update(detection, now) if not paused else False
                        if detection is not None and detection["contact"] is None and detection["detected"]:
                            rejected += 1
                        if due:
                            candidate = detection
                    if frame is not None:
                        cropped = fresh_crop
                        h, w = cropped.shape[:2]
                        scale = min(960 / w, 720 / h, 1.0)
                        display = cv2.resize(cropped, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
                        displayed += 1
                        elapsed = time.perf_counter() - fps_start
                        if elapsed >= 0.5:
                            display_fps = displayed / elapsed
                            displayed, fps_start = 0, time.perf_counter()
                        if recording:
                            state = "Paused" if paused else contact.status
                            controls = f"{captured}/{target} valid | {state} | P pause/resume | Q quit"
                            prediction = contact.last_detection
                            if not paused and contact.active and prediction["contact"] is not None:
                                x1, y1, x2, y2 = [round(value * scale) for value in prediction["box"]]
                                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        else:
                            controls = f"{captured}/{target} | SPACE save | T timed {'ON' if automatic else 'OFF'} | Q quit"
                        cv2.putText(display, controls,
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                        cv2.putText(display, f"Camera {camera.fps:.1f} | Display {display_fps:.1f} FPS | {camera.format}"
                                    + (" | Writer busy" if busy else ""),
                                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                        cv2.imshow(title, display)
                    key = cv2.pollKey() & 0xFF
                    if frame is None:
                        time.sleep(0.001)
                    if key in (27, ord("q")):
                        break
                    if cropped is not None and cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if recording and key == ord("p"):
                        paused = not paused
                        contact.reset(time.monotonic())
                        candidate = None
                        due = False
                    elif not recording and key == ord("t"):
                        automatic = not automatic
                    if recording:
                        save_frame = not paused and candidate is not None
                    else:
                        manual = key == ord(" ") and cropped is not None
                        timed = automatic and frame is not None and now - last_capture >= interval
                        save_frame = manual or timed
                    if save_frame:
                        if recording:
                            busy = not writer.submit(output / f"frame_{number + 1:06d}.png",
                                                     candidate["frame"], candidate["contact"])
                        else:
                            busy = not writer.submit(output / f"frame_{number + 1:06d}.png", cropped)
                        if not busy:
                            number += 1
                            captured += 1
                            last_capture = now
                            if recording:
                                contact.last_saved = candidate["timestamp"]
                                candidate = None
    finally:
        cv2.destroyAllWindows()
    print(f"Saved {captured} cropped images to {output}")
    if recording:
        print(f"Rejected {rejected} invalid contacts before saving. Accepted geometry: {output / 'contacts.json'}")


if __name__ == "__main__":
    main()
