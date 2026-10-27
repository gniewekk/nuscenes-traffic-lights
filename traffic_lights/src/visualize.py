"""
Load the FiftyOne JSON and open the App to check detections by eye.

In Docker set:
  FIFTYONE_DATABASE_URI=mongodb://mongo:27017
  FIFTYONE_DEFAULT_APP_ADDRESS=0.0.0.0
  FIFTYONE_DEFAULT_APP_PORT=5151
"""

import os
import json

import fiftyone as fo

os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

# Prefer env (compose); fall back to FiftyOne config / localhost
app_address = os.environ.get(
    "FIFTYONE_DEFAULT_APP_ADDRESS",
    getattr(fo.config, "default_app_address", None) or "127.0.0.1",
)
app_port = int(
    os.environ.get(
        "FIFTYONE_DEFAULT_APP_PORT",
        str(getattr(fo.config, "default_app_port", None) or 5151),
    )
)

with open("boxes_dict_for_fiftyone.json", "r") as f:
    boxes_dict_for_fiftyone = json.load(f)

dataset = fo.Dataset("traffic_lights", overwrite=True)
for file_path, list_of_lights in boxes_dict_for_fiftyone.items():
    sample = fo.Sample(filepath=os.path.abspath(file_path))
    fo_detections_list = []

    for light in list_of_lights:
        detection = fo.Detection(
            label=light["color"],
            bounding_box=light["box"],
            confidence=light["confidence"],
        )
        fo_detections_list.append(detection)
    sample["yolo_predictions"] = fo.Detections(detections=fo_detections_list)
    dataset.add_sample(sample)

fo.close_app()
session = fo.launch_app(dataset, address=app_address, port=app_port)
print(f"FiftyOne App: http://{app_address}:{app_port}")
session.wait()
