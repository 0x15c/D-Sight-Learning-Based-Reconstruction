"""Generate native-resolution targets from contacts already accepted at capture."""

import json
import shutil

import cv2
import numpy as np

from calibration import ball_height_map
from sensor_config import SensorConfig


def main():
    config = SensorConfig.select()
    paths = config.data["paths"]
    recordings = config.resolve(paths["recording_images"])
    contacts_path = recordings / "contacts.json"
    if not contacts_path.exists():
        raise RuntimeError("Record with the updated automatic capture first to create contacts.json; existing recordings are unchanged.")
    recording = json.loads(contacts_path.read_text(encoding="utf-8"))
    contacts = recording["samples"]
    if not contacts:
        raise RuntimeError("No accepted contacts yet. Record valid ball presses with record_calibration.py first.")
    output = config.resolve(paths["dataset"])
    for directory in ("images", "heights", "preview"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    samples = []
    height, width = recording["image_size"]
    # Geometry was measured for these exact frames, not the current camera setup.
    diameter, pitch = recording["ball_diameter_mm"], recording["mm_per_pixel"]
    print(f"Ball diameter: {diameter:g} mm; source pixel pitch: {pitch:g} mm/px")
    preview_stride = max(1, len(contacts) // 12)
    for index, contact in enumerate(contacts):
        path = recordings / contact["image"]
        frame = cv2.imread(str(path))
        if frame.shape[:2] != (height, width):
            raise RuntimeError("Saved image dimensions differ from contacts.json. Keep a fixed camera crop for each recording.")
        target = ball_height_map(height, width, contact["center_px"], contact["radius_px"], diameter, pitch)
        image_name, height_name = f"images/{path.name}", f"heights/{path.stem}.npy"
        shutil.copy2(path, output / image_name)
        np.save(output / height_name, target)
        samples.append({**contact, "image": image_name, "height": height_name})
        if index % preview_stride == 0:
            preview = frame.copy()
            cv2.circle(preview, tuple(round(x) for x in contact["center_px"]), round(contact["radius_px"]), (0, 255, 255), 2)
            preview_max = max(float(target.max()), 1e-8)
            heat = cv2.applyColorMap(np.uint8(np.clip(target[..., 0] / preview_max, 0, 1) * 255), cv2.COLORMAP_TURBO)
            cv2.putText(heat, f"depth color scale: 0 to {preview_max:.3f} mm",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            preview = np.concatenate([preview, heat], axis=1)
            if preview.shape[1] > 1600:
                preview = cv2.resize(preview, (1600, round(preview.shape[0] * 1600 / preview.shape[1])))
            cv2.imwrite(str(output / "preview" / path.name), preview)
        if (index + 1) % 50 == 0:
            print(f"{index+1}/{len(contacts)} accepted contacts converted")
    manifest = {**recording, "samples": samples, "units": "mm", "depth_convention": "positive_indentation"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved all {len(samples)} accepted contacts as {height}×{width}×1 targets. Review {output / 'preview'} before training.")


if __name__ == "__main__":
    main()
