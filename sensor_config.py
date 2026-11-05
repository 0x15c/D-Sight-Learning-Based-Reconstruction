"""One commented configuration per physical D-Sight sensor."""

import json
from pathlib import Path
import re
import tkinter as tk
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parent
SENSOR_DIR = ROOT / "sensors"
# Match strings before comments so paths and quoted URLs are left intact.
TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/|[^\s{}\[\],:/]+|[{}\[\],:]')


class SensorConfig:
    def __init__(self, path):
        self.path = Path(path)
        self.text = self.path.read_text(encoding="utf-8-sig")
        self.data, self.spans = self._parse(self.text)

    @staticmethod
    def _parse(text):
        """Read JSONC and retain value positions for comment-preserving saves."""
        tokens = [m for m in TOKENS.finditer(text) if not m.group().startswith(("//", "/*"))]
        spans = {}
        index = 0

        def value(path):
            nonlocal index
            start = tokens[index].start()
            token = tokens[index].group()
            index += 1
            if token == "{":
                result = {}
                while tokens[index].group() != "}":
                    key = json.loads(tokens[index].group())
                    index += 2  # key and colon
                    result[key] = value(path + (key,))
                    if tokens[index].group() == ",":
                        index += 1
                index += 1
            elif token == "[":
                result = []
                while tokens[index].group() != "]":
                    result.append(value(path + (len(result),)))
                    if tokens[index].group() == ",":
                        index += 1
                index += 1
            else:
                result = json.loads(token)
            spans[path] = (start, tokens[index - 1].end())
            return result

        return value(()), spans

    def resolve(self, value):
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    def save(self):
        original, spans = self._parse(self.text)
        replacements = []

        def update(old, new, path):
            if old == new:
                return
            if isinstance(old, dict) and isinstance(new, dict) and old.keys() == new.keys():
                for key in old:
                    update(old[key], new[key], path + (key,))
            elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
                for index, (before, after) in enumerate(zip(old, new)):
                    update(before, after, path + (index,))
            else:
                start, end = spans[path]
                replacements.append((start, end, json.dumps(new, ensure_ascii=False)))

        update(original, self.data, ())
        for start, end, replacement in sorted(replacements, reverse=True):
            self.text = self.text[:start] + replacement + self.text[end:]
        self.path.write_text(self.text, encoding="utf-8")

    @classmethod
    def create(cls, name, variant="markerless"):
        slug = re.sub(r"[^\w-]+", "_", name.strip()).strip("_") or "sensor"
        path = SENSOR_DIR / f"{slug}.jsonc"
        if path.exists():
            raise FileExistsError(f"Sensor '{slug}' already exists. Select it from the list.")
        prefix = f"data/{slug}"
        marked = variant == "marker"
        sections = {
            "sensor": ("Physical sensor identity. Presets are starting points; tune each sensor independently.",
                       {"name": name, "variant": variant}),
            "camera": ("Crop is [left, top, width, height] in the FULL camera frame. null means full frame.\n"
                       "Camera property units and auto-exposure values depend on the driver; DSHOW manual is usually 0.25.\n"
                       "A null setting leaves the camera driver's current value unchanged.\n"
                       "fourcc MJPG requests compressed USB capture; use an empty string for driver default.", {
                "index": 0, "backend": "DSHOW", "width": 1920, "height": 1080, "fps": 30, "fourcc": "MJPG",
                "settings": {"auto_exposure": 0.25, "exposure": -6, "brightness": None,
                             "contrast": None, "gain": None, "saturation": None,
                             "sharpness": None, "gamma": None, "auto_wb": 0,
                             "white_balance_temperature": None}, "crop": None}),
            "calibration": ("Measure mm_per_pixel in the native CROPPED image, before training resize.\n"
                            "Ball diameter and scale below are examples: replace them with your measured values.\n"
                            "Recording saves only frames with a valid YOLO contact. release_delay_s holds the UI through lift-off.\n"
                            "detection_fps limits background YOLO checks; capture_interval_s spaces saved frames.", {
                "ball_diameter_mm": 7.0, "mm_per_pixel": 34 / 1460, "annotation_count": 100,
                "record_count": 2000, "capture_interval_s": 0.25,
                "detection_fps": 10, "release_delay_s": 0.6}),
            "paths": ("Relative paths resolve from the reconstruction project folder. Scripts fill checkpoint paths.", {
                "annotation_images": f"{prefix}/annotation/images", "annotation_labels": f"{prefix}/annotation/labels",
                "recording_images": f"{prefix}/recording", "dataset": f"{prefix}/dataset",
                "background": f"{prefix}/background.png", "yolo_checkpoint": None, "checkpoint": None}),
            "yolo": ("One class: ball-contact footprint. image_size null uses native crop pixels plus stride padding.\n"
                     "Marked sensors start with a higher confidence threshold.", {
                "model": "yolo11n.pt", "epochs": 100, "image_size": None, "batch_size": 2,
                "confidence": 0.6 if marked else 0.5, "validation_fraction": 0.2}),
            "training": ("image_size null uses the native cropped [height, width]. Explicit sizes retain legacy resizing.\n"
                         "Depth is in mm; height reconstruction is the primary loss.\n"
                         "RAM cache is bounded by cache_limit_gib; larger datasets stream from disk.\n"
                         "The marker preset starts with a wider U-Net. Both presets remain editable.", {
                "image_size": None, "base_channels": 32 if marked else 16, "batch_size": 1,
                "epochs": 100, "learning_rate": 0.0001, "validation_fraction": 0.2,
                "seed": 42, "cache_limit_gib": 32, "num_workers": 0, "amp": True,
                "gradient_loss_weight": 0.0, "subtract_background": True}),
            "reconstruction": ("neumann integrates boundary slopes without forcing every image edge to zero.\n"
                               "reference_region is [left, top, width, height] in cropped source pixels.\n"
                               "Choose a known unloaded patch, or null to zero only the mean perimeter height.\n"
                               "This reference shifts one height offset; it does not flatten the perimeter.\n"
                               "Training saves these settings in the checkpoint; demo reuses them.\n"
                               "dirichlet retains the legacy fixed-zero perimeter solver.", {
                "boundary": "neumann", "reference_region": None}),
            "demo": ("Enable pyvista for a live 3D point cloud; min/max_depth_mm set the color scale.\n"
                     "min_depth_mm null uses -max_depth_mm for Neumann, or zero for legacy Dirichlet.", {
                "pyvista": False, "point_stride": 2, "max_depth_mm": 1.0,
                "min_depth_mm": None, "cloud_fps": 15}),
        }
        parts = []
        for key, (comment, value) in sections.items():
            note = "\n".join("  // " + line for line in comment.splitlines())
            body = json.dumps(value, indent=2, ensure_ascii=False).replace("\n", "\n  ")
            parts.append(f'{note}\n  "{key}": {body}')
        SENSOR_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text("{\n" + ",\n\n".join(parts) + "\n}\n", encoding="utf-8")
        return cls(path)

    @classmethod
    def select(cls, create=False):
        root = tk.Tk()
        root.title("Select D-Sight sensor")
        root.resizable(False, False)
        frame = ttk.Frame(root, padding=18)
        frame.grid()
        files = sorted(SENSOR_DIR.glob("*.jsonc"))
        ttk.Label(frame, text="Choose the physical sensor used for this session.").grid(row=0, column=0, columnspan=2, sticky="w")
        selected = ttk.Combobox(frame, values=[p.stem for p in files], state="readonly", width=36)
        selected.grid(row=1, column=0, padx=(0, 8), pady=12)
        if files:
            selected.current(0)
        result = []

        def choose():
            if selected.current() >= 0:
                result.append(cls(files[selected.current()]))
                root.destroy()

        ttk.Button(frame, text="Open sensor", command=choose).grid(row=1, column=1)
        if create:
            ttk.Separator(frame).grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)
            ttk.Label(frame, text="Create a new sensor").grid(row=3, column=0, sticky="w")
            name = ttk.Entry(frame, width=39)
            name.grid(row=4, column=0, padx=(0, 8), pady=8)
            variant = ttk.Combobox(frame, values=["markerless", "marker"], state="readonly", width=12)
            variant.current(0)
            variant.grid(row=4, column=1)

            def new_sensor():
                if not name.get().strip():
                    return
                try:
                    result.append(cls.create(name.get().strip(), variant.get()))
                except FileExistsError as error:
                    messagebox.showinfo("Sensor already exists", str(error), parent=root)
                    return
                root.destroy()

            ttk.Button(frame, text="Create sensor", command=new_sensor).grid(row=5, column=1, pady=6)
        elif not files:
            ttk.Label(frame, text="Run camera_setup.py first to create a sensor.").grid(row=2, column=0, columnspan=2)
        root.mainloop()
        if not result:
            raise SystemExit("Sensor selection cancelled.")
        return result[0]
