"""
Pull CAM_FRONT image paths from nuScenes mini and dump them to JSON.
"""

import os
import json

from nuscenes.nuscenes import NuScenes

# repo root (traffic_lights/src → ../..)
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

nusc = NuScenes(version="v1.0-mini", dataroot="./data/nuscenes/sets/v1.0-mini/", verbose=True)

paths_for_samples = []

for sample in nusc.sample:
    token = sample["data"]["CAM_FRONT"]
    filepath = nusc.get_sample_data(token)[0]

    todict = {
        "sample_token": sample["token"],
        "cam_front_token": token,
        "filepath": filepath,
    }
    paths_for_samples.append(todict)

with open("paths_for_samples.json", "w") as file:
    json.dump(paths_for_samples, file, indent=4)
