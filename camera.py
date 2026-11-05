"""Continuous camera capture with a single latest-frame slot."""

import threading
import time
import cv2


CAMERA_PROPERTIES = {
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "gain": cv2.CAP_PROP_GAIN,
    "saturation": cv2.CAP_PROP_SATURATION,
    "sharpness": cv2.CAP_PROP_SHARPNESS,
    "gamma": cv2.CAP_PROP_GAMMA,
    "auto_wb": cv2.CAP_PROP_AUTO_WB,
    "white_balance_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
}


class Camera:
    def __init__(self, config):
        self.config = config
        settings = config.data["camera"]
        backend = getattr(cv2, "CAP_" + settings["backend"])
        self.cap = cv2.VideoCapture(settings["index"], backend)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Could not open camera {settings['index']} with {settings['backend']}.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings["height"])
        self.cap.set(cv2.CAP_PROP_FPS, settings["fps"])
        # Uncompressed HD modes can be limited by USB bandwidth. An empty string
        # leaves format selection to the driver; old configs use MJPG by default.
        fourcc = settings.get("fourcc", "MJPG")
        self.format_accepted = True
        if fourcc:
            self.format_accepted = self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Apply auto modes before their manual values.
        self.accepted = {}
        for name in CAMERA_PROPERTIES:
            value = settings["settings"].get(name)
            if value is not None:
                self.accepted[name] = self.cap.set(CAMERA_PROPERTIES[name], value)
        code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.format = "".join(chr((code >> (8 * i)) & 255) for i in range(4)).strip("\x00") or "driver default"
        self.fps = 0.0
        self._capture_lock = threading.Lock()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._frame = None
        self._sequence = self._consumed = 0
        self._error = None

    def _capture(self):
        start, count = time.perf_counter(), 0
        try:
            while not self._stop.is_set():
                with self._capture_lock:
                    success, frame = self.cap.read()
                if not success:
                    raise RuntimeError("Camera stopped returning frames.")
                count += 1
                elapsed = time.perf_counter() - start
                if elapsed >= 0.5:
                    self.fps = count / elapsed
                    start, count = time.perf_counter(), 0
                with self._condition:
                    self._frame = frame
                    self._sequence += 1
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()
        finally:
            with self._capture_lock:
                self.cap.release()

    def read(self, timeout=None):
        """Return the newest unread BGR frame; timeout=0 polls without blocking.

        Slow consumers skip obsolete preview frames instead of accumulating lag.
        The returned image must not be modified in place.
        """
        with self._condition:
            if self._thread is None and not self._stop.is_set():
                self._thread = threading.Thread(target=self._capture, daemon=True, name="D-Sight camera")
                self._thread.start()
            self._condition.wait_for(lambda: self._sequence > self._consumed or self._error or self._stop.is_set(), timeout)
            if self._sequence > self._consumed:
                self._consumed = self._sequence
                return self._frame
            if self._error:
                raise RuntimeError(str(self._error)) from self._error
            return None

    def set(self, prop, value):
        with self._capture_lock:
            return self.cap.set(prop, value)

    def get(self, prop):
        with self._capture_lock:
            return self.cap.get(prop)

    def crop(self, frame):
        crop = self.config.data["camera"]["crop"]
        if crop is None:
            return frame
        x, y, width, height = crop
        return frame[y:y + height, x:x + width]

    def close(self):
        if self._stop.is_set():
            return
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is None:
            with self._capture_lock:
                self.cap.release()
        else:
            # The capture thread owns release, including on disconnect. Avoid
            # racing release against a backend that is still inside read().
            self._thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
