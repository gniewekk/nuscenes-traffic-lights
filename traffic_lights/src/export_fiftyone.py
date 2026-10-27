"""
Rewrite absolute xyxy boxes into FiftyOne's relative [x, y, w, h] format.
"""

import os
import json
import collections

os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 928

with open("boxes_dict.json", "r") as f:
    file = json.load(f)
    boxes_dict_for_fiftyone = collections.defaultdict(list)
    for instance in file:
        filepath = instance["filepath"]
        xmin, ymin, xmax, ymax = instance["box"]
        box_width = (xmax - xmin) / IMAGE_WIDTH
        box_height = (ymax - ymin) / IMAGE_HEIGHT
        top_left_x = xmin / IMAGE_WIDTH
        top_left_y = ymin / IMAGE_HEIGHT
        new_box = [top_left_x, top_left_y, box_width, box_height]

        boxes_dict_for_fiftyone[filepath].append(
            {
                "box": new_box,
                "color": instance["color"],
                "confidence": instance["confidence"],
            }
        )

with open("boxes_dict_for_fiftyone.json", "w") as f:
    json.dump(boxes_dict_for_fiftyone, f, indent=4)
