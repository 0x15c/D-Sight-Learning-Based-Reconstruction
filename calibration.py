"""Ball geometry in cropped camera pixels and millimetres."""

import numpy as np


def box_to_circle(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2, ((x2 - x1) + (y2 - y1)) / 4


def ball_height_map(height, width, center_px, radius_px, ball_diameter_mm, mm_per_pixel):
    """Positive indentation of a sphere above its circular contact plane, H×W×1.

    mm_per_pixel is measured at the ORIGINAL cropped camera resolution. The
    detected contact radius is not the physical ball radius.
    """
    ball_radius = ball_diameter_mm / 2
    contact_radius = radius_px * mm_per_pixel
    if not 0 < contact_radius < ball_radius:
        raise ValueError("Contact radius must be smaller than the physical ball radius; check mm_per_pixel.")
    yy, xx = np.ogrid[:height, :width]
    distance_squared = ((xx - center_px[0]) ** 2 + (yy - center_px[1]) ** 2) * mm_per_pixel**2
    inside = distance_squared <= contact_radius**2
    depth = np.zeros((height, width, 1), dtype=np.float32)
    depth[..., 0][inside] = (
        np.sqrt(ball_radius**2 - distance_squared[inside])
        - np.sqrt(ball_radius**2 - contact_radius**2)
    )
    return depth


def select_contact(boxes, confidence, height, width, ball_diameter_mm, mm_per_pixel, margin=1):
    """Highest-confidence interior, nearly square, physically possible contact."""
    for index in np.argsort(confidence)[::-1]:
        x1, y1, x2, y2 = boxes[index]
        box_width, box_height = x2 - x1, y2 - y1
        if min(box_width, box_height) <= 0:
            continue
        if max(box_width, box_height) / min(box_width, box_height) > 1.3:
            continue
        cx, cy, radius = box_to_circle(boxes[index])
        # Contact must remain inside the image; a clipped box cannot define a cap.
        if min(x1, y1, cx - radius, cy - radius) < margin:
            continue
        if max(x2, cx + radius) >= width - margin or max(y2, cy + radius) >= height - margin:
            continue
        if radius * mm_per_pixel >= ball_diameter_mm / 2:
            continue
        return {"center_px": [float(cx), float(cy)], "radius_px": float(radius), "confidence": float(confidence[index])}
    return None
