"""Run directly: choose a sensor, adjust its camera, drag crop corners, save."""

import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from camera import CAMERA_PROPERTIES, Camera
from sensor_config import SensorConfig


class CameraSetup:
    def __init__(self, config):
        self.config = config
        self.camera = None
        self.frame = None
        self.corner = None
        self.rectangle = None
        self.scale = 1.0
        self.preview_size = None
        self.display_count = 0
        self.display_started = time.perf_counter()
        self.root = tk.Tk()
        self.root.title(f"D-Sight camera setup — {config.data['sensor']['name']}")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas = tk.Canvas(self.root, width=960, height=600, background="#202020", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", lambda event: setattr(self, "corner", None))
        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=1, sticky="ns")
        ttk.Label(controls, text="Camera connection").grid(row=0, column=0, columnspan=2, sticky="w")
        self.connection = {}
        for row, name in enumerate(("index", "backend", "width", "height", "fps", "fourcc"), 1):
            ttk.Label(controls, text=name).grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value=str(config.data["camera"].get(name, "MJPG")))
            ttk.Entry(controls, textvariable=var, width=12).grid(row=row, column=1, pady=2)
            self.connection[name] = var
        ttk.Button(controls, text="Open / reconnect", command=self.connect).grid(row=7, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(controls, text="Driver settings (numeric)").grid(row=8, column=0, columnspan=2, sticky="w")
        self.properties = {}
        for row, name in enumerate(CAMERA_PROPERTIES, 9):
            ttk.Label(controls, text=name).grid(row=row, column=0, sticky="w")
            value = config.data["camera"]["settings"].get(name)
            var = tk.StringVar(value="" if value is None else str(value))
            ttk.Entry(controls, textvariable=var, width=12).grid(row=row, column=1, pady=2)
            self.properties[name] = var
        ttk.Button(controls, text="Apply settings", command=self.apply_settings).grid(row=19, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(controls, text="Drag any yellow corner to crop.\nBlank setting: leave driver unchanged.\nDSHOW manual exposure: usually 0.25.\nMJPG usually enables faster USB capture.").grid(row=20, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Button(controls, text="Reset crop to full frame", command=self.reset_crop).grid(row=21, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(controls, text="Save settings + crop", command=self.save).grid(row=22, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(controls, text="Save unloaded background", command=self.save_background).grid(row=23, column=0, columnspan=2, sticky="ew", pady=3)
        self.fps_text = tk.StringVar(value="Waiting for camera frames...")
        ttk.Label(self.root, textvariable=self.fps_text, padding=(8, 4)).grid(row=1, column=0, columnspan=2, sticky="w")
        self.status = tk.StringVar(value="Remove all contact before saving an unloaded background.")
        ttk.Label(self.root, textvariable=self.status, padding=8, wraplength=1100).grid(row=2, column=0, columnspan=2, sticky="w")
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.crop_outline = self.canvas.create_rectangle(0, 0, 0, 0, outline="#ffe55b", width=2)
        self.crop_handles = [self.canvas.create_rectangle(0, 0, 0, 0, fill="#ffe55b", outline="#202020") for _ in range(4)]
        self.connect()
        self.root.after(1, self.update_frame)

    def connect(self):
        if self.camera:
            self.camera.close()
            self.camera = None
        settings = self.config.data["camera"]
        for name, var in self.connection.items():
            settings[name] = var.get().strip().upper() if name in ("backend", "fourcc") else int(var.get())
        self.rectangle = None
        self.frame = None
        self.preview_size = None
        self.display_count = 0
        self.display_started = time.perf_counter()
        try:
            self.camera = Camera(self.config)
            for name, var in self.properties.items():
                if not var.get():
                    var.set(f"{self.camera.get(CAMERA_PROPERTIES[name]):g}")
            rejected = [name for name, accepted in self.camera.accepted.items() if not accepted]
            if not self.camera.format_accepted:
                rejected.append(f"fourcc={settings['fourcc']}")
            self.status.set("Camera opened. " + ("Driver rejected: " + ", ".join(rejected) if rejected else "Drag yellow corners to crop."))
        except RuntimeError as error:
            self.status.set(str(error) + " Edit the connection above and reconnect.")

    def apply_settings(self):
        if self.camera is None:
            return
        rejected = []
        actual = []
        for name, var in self.properties.items():
            value = float(var.get()) if var.get().strip() else None
            if value is None:
                self.config.data["camera"]["settings"][name] = None
                continue
            if self.camera.set(CAMERA_PROPERTIES[name], value):
                # Save the request: some drivers return meaningless readback values.
                self.config.data["camera"]["settings"][name] = value
                actual.append(f"{name}={self.camera.get(CAMERA_PROPERTIES[name]):g}")
            else:
                rejected.append(name)
        self.status.set("Readback: " + ", ".join(actual) + (". Driver rejected: " + ", ".join(rejected) if rejected else ""))

    def update_frame(self):
        if self.camera:
            try:
                frame = self.camera.read(timeout=0)
            except RuntimeError as error:
                self.status.set(str(error))
                self.camera.close()
                self.camera = None
            else:
                if frame is None:
                    self.root.after(1, self.update_frame)
                    return
                self.frame = frame
                height, width = self.frame.shape[:2]
                self.scale = min(960 / width, 720 / height, 1.0)
                size = (round(width * self.scale), round(height * self.scale))
                display = cv2.resize(self.frame, size, interpolation=cv2.INTER_AREA)
                image = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                # Reuse Tk storage; PNG encoding/decoding used to dominate this loop.
                if size != self.preview_size:
                    self.preview_size = size
                    self.photo = ImageTk.PhotoImage(image)
                    self.canvas.configure(width=size[0], height=size[1])
                    self.canvas.itemconfigure(self.image_item, image=self.photo)
                else:
                    self.photo.paste(image)
                if self.rectangle is None:
                    crop = self.config.data["camera"]["crop"] or [0, 0, width, height]
                    x, y, w, h = crop
                    # A new camera mode may be smaller than the previous crop.
                    if x + w > width or y + h > height:
                        x, y, w, h = 0, 0, width, height
                    self.rectangle = [x, y, x + w, y + h]
                    self.draw_crop()
                self.display_count += 1
                now = time.perf_counter()
                elapsed = now - self.display_started
                if elapsed >= 0.5:
                    self.fps_text.set(f"Camera {self.camera.fps:.1f} FPS | Display {self.display_count / elapsed:.1f} FPS | {self.camera.format} | {width} x {height}")
                    self.display_started = now
                    self.display_count = 0
        # Poll the latest-frame slot without blocking Tk or adding a frame's delay.
        self.root.after(1 if self.camera else 30, self.update_frame)

    def draw_crop(self):
        x1, y1, x2, y2 = [value * self.scale for value in self.rectangle]
        self.canvas.coords(self.crop_outline, x1, y1, x2, y2)
        for handle, (x, y) in zip(self.crop_handles, ((x1, y1), (x2, y1), (x2, y2), (x1, y2))):
            self.canvas.coords(handle, x - 6, y - 6, x + 6, y + 6)

    def start_drag(self, event):
        if self.rectangle is None:
            return
        x1, y1, x2, y2 = self.rectangle
        corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        distances = [(x * self.scale - event.x) ** 2 + (y * self.scale - event.y) ** 2 for x, y in corners]
        nearest = min(range(4), key=distances.__getitem__)
        self.corner = nearest if distances[nearest] < 24 ** 2 else None

    def drag(self, event):
        if self.corner is None:
            return
        height, width = self.frame.shape[:2]
        x = min(width, max(0, round(event.x / self.scale)))
        y = min(height, max(0, round(event.y / self.scale)))
        x1, y1, x2, y2 = self.rectangle
        if self.corner in (0, 3):
            x1 = min(x, x2 - 8)
        else:
            x2 = max(x, x1 + 8)
        if self.corner in (0, 1):
            y1 = min(y, y2 - 8)
        else:
            y2 = max(y, y1 + 8)
        self.rectangle = [x1, y1, x2, y2]
        self.draw_crop()

    def reset_crop(self):
        if self.frame is not None:
            height, width = self.frame.shape[:2]
            self.rectangle = [0, 0, width, height]
            self.draw_crop()

    def save(self):
        if self.frame is None:
            return
        self.apply_settings()
        x1, y1, x2, y2 = self.rectangle
        self.config.data["camera"]["crop"] = [x1, y1, x2 - x1, y2 - y1]
        height, width = self.frame.shape[:2]
        self.config.data["camera"].update(width=width, height=height)
        self.config.save()
        self.status.set(f"Saved {self.config.path.name}: crop {x2-x1} × {y2-y1}. Re-save the unloaded background after changing crop/settings.")

    def save_background(self):
        if self.frame is None:
            return
        if not messagebox.askokcancel("Unloaded background", "Remove the calibration ball and all contact from the sensor, then press OK.", parent=self.root):
            return
        self.save()
        self.frame = self.camera.read()
        path = self.config.resolve(self.config.data["paths"]["background"])
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.camera.crop(self.frame))
        self.status.set(f"Saved unloaded cropped background: {path.name}")

    def close(self):
        if self.camera:
            self.camera.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    CameraSetup(SensorConfig.select(create=True)).run()
