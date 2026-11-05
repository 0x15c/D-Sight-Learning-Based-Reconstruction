"""Train one-class YOLO from the selected sensor's manually annotated images."""

from datetime import datetime
import json
import random
import shutil

import cv2
import torch

from sensor_config import SensorConfig
from yolo_resolution import pad_native_image


def main():
    config = SensorConfig.select()
    from ultralytics import YOLO
    settings = config.data["yolo"]
    images = config.resolve(config.data["paths"]["annotation_images"])
    labels = config.resolve(config.data["paths"]["annotation_labels"])
    annotated = [path for path in sorted(images.glob("*.png")) if (labels / (path.stem + ".txt")).exists()]
    if len(annotated) < 2:
        raise RuntimeError("Save annotations for at least two images with annotate_ball.py first.")
    net = YOLO(settings["model"])
    native = settings["image_size"] is None
    image_size = settings["image_size"]
    if native:
        stride = max(int(net.model.stride.max()), 32)
        original_shape = cv2.imread(str(annotated[0])).shape[:2]
        padded_shape = tuple(((size + stride - 1) // stride) * stride for size in original_shape)
        image_size = max(padded_shape)
        print(f"Native YOLO training: {original_shape[1]} x {original_shape[0]} pixels; "
              f"stride padding to {padded_shape[1]} x {padded_shape[0]} (no resampling).")
    seed = config.data["training"]["seed"]
    random.Random(seed).shuffle(annotated)
    validation_count = max(1, min(len(annotated) - 1, round(len(annotated) * settings["validation_fraction"])))
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = config.resolve(f"runs/{config.path.stem}/detector/{run_name}")
    dataset = run_root / "dataset"
    # A fresh, small detector dataset avoids stale split files on later runs.
    for split, paths in (("val", annotated[:validation_count]), ("train", annotated[validation_count:])):
        image_output, label_output = dataset / "images" / split, dataset / "labels" / split
        image_output.mkdir(parents=True)
        label_output.mkdir(parents=True)
        for path in paths:
            source_label = labels / (path.stem + ".txt")
            destination_label = label_output / (path.stem + ".txt")
            if native:
                frame = cv2.imread(str(path))
                if frame.shape[:2] != original_shape:
                    raise RuntimeError("Annotation dimensions changed. Use one fixed camera crop per sensor.")
                cv2.imwrite(str(image_output / path.name), pad_native_image(frame, stride))
                # Padding changes the normalized label denominators, not pixel coordinates.
                h, w = original_shape
                padded_h, padded_w = padded_shape
                converted = []
                for row in source_label.read_text().splitlines():
                    cls, x, y, bw, bh = row.split()
                    converted.append(f"{cls} {float(x)*w/padded_w:.10f} {float(y)*h/padded_h:.10f} "
                                     f"{float(bw)*w/padded_w:.10f} {float(bh)*h/padded_h:.10f}")
                destination_label.write_text("\n".join(converted) + ("\n" if converted else ""))
            else:
                shutil.copy2(path, image_output / path.name)
                shutil.copy2(source_label, destination_label)
    yaml_path = dataset / "data.yaml"
    # JSON is valid YAML, and safely quotes Windows paths.
    yaml_path.write_text(json.dumps({"path": dataset.as_posix(), "train": "images/train", "val": "images/val",
                                     "names": {0: "ball_contact"}}, indent=2), encoding="utf-8")
    # Ultralytics first resizes the long side to scalar imgsz. Pre-padding makes
    # that ratio exactly one; rectangular batches then retain the native pixels.
    native_options = dict(rect=True, scale=0.0, translate=0.0, degrees=0.0,
                          shear=0.0, perspective=0.0, multi_scale=0.0) if native else {}
    net.train(data=str(yaml_path), epochs=settings["epochs"], imgsz=image_size,
              batch=settings["batch_size"], device=0 if torch.cuda.is_available() else "cpu",
              project=str(run_root), name="train", workers=0, seed=seed, cache=False,
              hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.0, flipud=0.0, mosaic=0.0,
              **native_options)
    config.data["paths"]["yolo_checkpoint"] = str(net.trainer.best.resolve())
    config.save()
    print(f"Detector checkpoint saved in {config.path}: {net.trainer.best}")


if __name__ == "__main__":
    main()
