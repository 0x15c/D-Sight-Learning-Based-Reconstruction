"""Run the selected sensor's ball detector without blocking its live preview."""

from concurrent.futures import ThreadPoolExecutor
import math

from calibration import select_contact
from yolo_resolution import pad_native_image


class BallDetector:
    def __init__(self, config):
        checkpoint = config.data["paths"]["yolo_checkpoint"]
        if not checkpoint:
            raise RuntimeError("Train this sensor's ball detector with train_detector.py before automatic recording.")
        self.checkpoint = str(config.resolve(checkpoint))
        self.image_size = config.data["yolo"]["image_size"]
        self.confidence = config.data["yolo"]["confidence"]
        self.calibration = config.data["calibration"]
        self.training_size = config.data["training"]["image_size"]
        self.interval = 1.0 / config.data["calibration"].get("detection_fps", 10)
        self.next_detection = float("-inf")
        self.model = None
        self.future = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="D-Sight YOLO")
        self.closed = False

    def _predict(self, frame, timestamp):
        # Loading and all predictor access happen on the same worker. Annotation
        # capture never imports these libraries or downloads a generic detector.
        if self.model is None:
            import torch
            from ultralytics import YOLO

            self.device = 0 if torch.cuda.is_available() else "cpu"
            self.model = YOLO(self.checkpoint)
            self.stride = max(int(self.model.model.stride.max()), 32)
        height, width = frame.shape[:2]
        image = pad_native_image(frame, self.stride) if self.image_size is None else frame
        image_size = list(image.shape[:2]) if self.image_size is None else self.image_size
        prediction = self.model.predict(
            image, imgsz=image_size, rect=True, conf=self.confidence, classes=[0],
            device=self.device, verbose=False,
        )[0]
        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            return {"timestamp": timestamp, "detected": False, "box": None, "confidence": 0.0,
                    "contact": None, "frame": frame}
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidence = boxes.conf.detach().cpu().numpy()
        train_height, train_width = self.training_size or (height, width)
        margin = max(1, math.ceil(max(height / train_height, width / train_width))) + 1
        contact = select_contact(xyxy, confidence, height, width,
                                 self.calibration["ball_diameter_mm"], self.calibration["mm_per_pixel"], margin)
        # Derive the overlay from the accepted circle, not an invalid higher-score box.
        if contact is not None:
            cx, cy = contact["center_px"]
            radius = contact["radius_px"]
            box = [cx - radius, cy - radius, cx + radius, cy + radius]
            score = contact["confidence"]
        else:
            index = int(confidence.argmax())
            box, score = xyxy[index].tolist(), float(confidence[index])
        return {
            "timestamp": timestamp,
            "detected": True,
            "box": box,
            "confidence": score,
            "contact": contact,
            "frame": frame,
        }

    def update(self, frame, timestamp):
        """Poll a finished result and, when idle, submit only the current frame.

        Results retain their source-frame timestamp, so old detections cannot
        extend a recording just because model loading or inference was slow.
        There is at most one inference in flight and no pending image backlog.
        """
        result = None
        if self.future is not None and self.future.done():
            completed, self.future = self.future, None
            result = completed.result()
        if not self.closed and frame is not None and self.future is None and timestamp >= self.next_detection:
            self.future = self.pool.submit(self._predict, frame.copy(), timestamp)
            self.next_detection = timestamp + self.interval
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
