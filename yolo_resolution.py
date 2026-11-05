"""Keep every cropped camera pixel while meeting YOLO's stride requirement."""

import cv2


def pad_native_image(frame, stride=32):
    """Pad only bottom/right, so box coordinates keep the crop's original origin."""
    height, width = frame.shape[:2]
    bottom, right = (-height) % stride, (-width) % stride
    if bottom == 0 and right == 0:
        return frame
    return cv2.copyMakeBorder(frame, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
