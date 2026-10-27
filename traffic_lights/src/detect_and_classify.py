"""
YOLOv8x finds traffic lights, then HSV heuristics decide Red/Yellow/Green/Unknown.
Unknown crops go to debug_unknown/ for a quick look later.
"""

import os
import json

import cv2
import numpy as np
from ultralytics import YOLO

os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

with open("paths_for_samples.json", "r") as f:
    paths_for_samples = json.load(f)

model = YOLO("yolov8x.pt")
print(model.names)  # 9: 'traffic light'

# HSV ranges tuned for camera footage (greens look cyan-ish, LED centers blow out)
color_bounds = {
    "Red": [
        (np.array([0, 0, 100]), np.array([10, 255, 255])),
        (np.array([170, 0, 100]), np.array([179, 255, 255])),
    ],
    "Yellow": [(np.array([13, 0, 100]), np.array([40, 255, 255]))],
    "Green": [(np.array([50, 0, 100]), np.array([100, 255, 255]))],
}


def classify_traffic_light_color(cropped_bgr, minimal_pixels=0.05):
    hsv_picture = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2HSV)
    scores = {}
    y, x = cropped_bgr.shape[:2]
    total_pixels = x * y
    if total_pixels == 0:
        return "Unknown"
    for color, boundaries in color_bounds.items():
        combined_mask = np.zeros(cropped_bgr.shape[:2], dtype=np.uint8)

        for lower_bound, upper_bound in boundaries:
            current_mask = cv2.inRange(hsv_picture, lower_bound, upper_bound)
            combined_mask = cv2.bitwise_or(combined_mask, current_mask)

        active_pixels = cv2.countNonZero(combined_mask)
        scores[color] = active_pixels

    best_color = max(scores, key=scores.get)
    max_pixels = scores[best_color]
    if max_pixels / total_pixels < minimal_pixels:
        return "Unknown"

    return best_color


os.makedirs("debug_unknown", exist_ok=True)
result_list = []

for sample in paths_for_samples:
    path = sample["filepath"]
    results = model.predict(source=path, conf=0.35, classes=[9], imgsz=1600)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # [xmin, ymin, xmax, ymax]
        confidence = result.boxes.conf.cpu().numpy()

        for idx, (b, c) in enumerate(zip(boxes, confidence)):
            xmin, ymin, xmax, ymax = b

            cropped = result.orig_img[
                max(0, int(ymin)) : min(result.orig_img.shape[0], int(ymax)),
                max(0, int(xmin)) : min(result.orig_img.shape[1], int(xmax)),
            ]

            color = classify_traffic_light_color(cropped)

            if color == "Unknown":
                cv2.imwrite(f"debug_unknown/{sample['sample_token']}_{idx}.jpg", cropped)

            result_dict = {
                "sample_token": sample["sample_token"],
                "cam_front_token": sample["cam_front_token"],
                "filepath": sample["filepath"],
                "box": b.tolist(),
                "color": color,
                "confidence": float(c),
            }
            result_list.append(result_dict)

with open("boxes_dict.json", "w") as f:
    json.dump(result_list, f, indent=4)
