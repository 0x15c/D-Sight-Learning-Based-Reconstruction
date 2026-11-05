"""Square contact annotations: drag a new square or one of its corners."""

import cv2
import numpy as np

from sensor_config import SensorConfig


class BallAnnotator:
    def __init__(self, config):
        self.images = sorted(config.resolve(config.data["paths"]["annotation_images"]).glob("*.png"))
        self.labels = config.resolve(config.data["paths"]["annotation_labels"])
        self.labels.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.box = None
        self.drag = None
        self.status = ""
        self.title = "Ball contact annotation"

    def load(self):
        self.frame = cv2.imread(str(self.images[self.index]))
        self.height, self.width = self.frame.shape[:2]
        self.scale = min(1.0, 1100 / self.width, 750 / self.height)
        self.box = None
        label = self.labels / (self.images[self.index].stem + ".txt")
        self.status = "Not annotated"
        if label.exists():
            self.status = "Saved"
            rows = label.read_text().strip().splitlines()
            if rows:
                _, cx, cy, width, height = map(float, rows[0].split())
                self.box = np.array([(cx - width / 2) * self.width, (cy - height / 2) * self.height,
                                     (cx + width / 2) * self.width, (cy + height / 2) * self.height])
        self.drag = None
        self.preview = cv2.resize(self.frame, None, fx=self.scale, fy=self.scale)
        self.preview = cv2.copyMakeBorder(self.preview, 0, 80, 0, 0, cv2.BORDER_CONSTANT)
        self.redraw = True

    def mouse(self, event, x, y, flags, param):
        point = np.array([x / self.scale, y / self.scale])
        point = np.clip(point, [0, 0], [self.width - 1, self.height - 1])
        if event == cv2.EVENT_LBUTTONDOWN:
            self.redraw = True
            self.status = "Unsaved"
            self.drag = ("corner", point)
            if self.box is not None:
                x1, y1, x2, y2 = self.box
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                nearest = np.argmin(np.linalg.norm(corners - point, axis=1))
                if np.linalg.norm(corners[nearest] - point) * self.scale < 14:
                    self.drag = ("corner", corners[(nearest + 2) % 4])
                elif x1 < point[0] < x2 and y1 < point[1] < y2:
                    self.drag = ("move", point, self.box.copy())
        if event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            self.redraw = True
            if self.drag[0] == "move":
                _, start, old_box = self.drag
                delta = np.clip(point - start, -old_box[:2], [self.width - 1 - old_box[2], self.height - 1 - old_box[3]])
                self.box = old_box + np.tile(delta, 2)
            else:
                anchor = self.drag[1]
                direction = np.where(point >= anchor, 1, -1)
                limit = np.where(direction > 0, [self.width - 1, self.height - 1] - anchor, anchor)
                side = min(np.max(np.abs(point - anchor)), np.min(limit))
                opposite = anchor + direction * side
                self.box = np.concatenate([np.minimum(anchor, opposite), np.maximum(anchor, opposite)])
        if event == cv2.EVENT_LBUTTONUP:
            self.drag = None

    def save(self):
        self.redraw = True
        label = self.labels / (self.images[self.index].stem + ".txt")
        if self.box is None:
            label.write_text("")  # Explicitly saved empty frame = negative example.
        else:
            x1, y1, x2, y2 = self.box
            if min(x2 - x1, y2 - y1) < 2:
                self.status = "Draw a larger contact box before saving"
                return False
            label.write_text(f"0 {(x1+x2)/(2*self.width):.8f} {(y1+y2)/(2*self.height):.8f} "
                             f"{(x2-x1)/self.width:.8f} {(y2-y1)/self.height:.8f}\n")
        self.status = "Saved"
        return True

    def run(self):
        if not self.images:
            raise RuntimeError("Record annotation images first with record_calibration.py.")
        self.load()
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.title, self.mouse)
        while True:
            if self.redraw:
                display = self.preview.copy()
                if self.box is not None:
                    x1, y1, x2, y2 = np.rint(self.box * self.scale).astype(int)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    for point in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                        cv2.circle(display, point, 5, (0, 255, 255), -1)
                lines = [f"{self.index+1}/{len(self.images)} {self.images[self.index].name} | {self.status}",
                         "SPACE save + next | S save | N/P next/previous (no save) | D clear | Q quit",
                         "Drag a corner to resize; drag inside to move. Save an empty box only for a no-contact image."]
                for row, text in enumerate(lines):
                    cv2.putText(display, text, (8, display.shape[0] - 58 + 23 * row),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(self.title, display)
                self.redraw = False
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                self.save()
            if key == ord("d"):
                self.box, self.status = None, "Unsaved empty annotation"
                self.redraw = True
            if key == ord(" ") and self.save():
                if self.index == len(self.images) - 1:
                    break
                self.index += 1
                self.load()
            elif key in (ord("n"), ord("p")):
                self.index = max(0, min(len(self.images) - 1, self.index + (1 if key == ord("n") else -1)))
                self.load()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    BallAnnotator(SensorConfig.select()).run()
