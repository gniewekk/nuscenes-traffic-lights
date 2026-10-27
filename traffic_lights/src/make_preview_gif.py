"""
Pick ~10 diverse detection frames from boxes_dict.json, draw boxes, write a GIF.

Usage (from repo root):
  python traffic_lights/src/make_preview_gif.py
  # → assets/predictions_preview.gif
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

BOXES_PATH = "boxes_dict.json"
OUT_PATH = Path("assets/predictions_preview.gif")
N_FRAMES = 10
FRAME_WIDTH = 800
DURATION_MS = 1200
MIN_TS_GAP = 3_000_000  # ~3 s within the same log
MAX_PER_SCENE = 2

# BGR for OpenCV drawing
COLOR_BGR = {
    "Red": (40, 40, 220),
    "Yellow": (0, 200, 255),
    "Green": (60, 180, 60),
    "Unknown": (160, 160, 160),
}


def scene_key(path: str) -> str:
    return Path(path).name.split("__")[0]


def timestamp(path: str) -> int:
    return int(Path(path).stem.split("__")[-1])


def box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def score_image(dets: list[dict]) -> float:
    colors = {d["color"] for d in dets}
    max_conf = max(d["confidence"] for d in dets)
    mean_conf = sum(d["confidence"] for d in dets) / len(dets)
    max_area = max(box_area(d["box"]) for d in dets)
    score = (
        len(colors) * 2.5
        + max_conf * 3
        + mean_conf
        + min(max_area / 800, 4)
        + min(len(dets), 4) * 0.3
    )
    if colors == {"Unknown"}:
        score -= 2
    return score


def select_paths(by_path: dict[str, list[dict]], n: int = N_FRAMES) -> list[str]:
    scored = sorted(
        ((score_image(dets), path) for path, dets in by_path.items()),
        reverse=True,
    )
    picked: list[str] = []
    used_scenes: collections.Counter = collections.Counter()

    def try_add(path: str, enforce_scene_cap: bool) -> bool:
        if not Path(path).exists():
            return False
        sk = scene_key(path)
        if enforce_scene_cap and used_scenes[sk] >= MAX_PER_SCENE:
            return False
        t = timestamp(path)
        for prev in picked:
            if scene_key(prev) == sk and abs(t - timestamp(prev)) < MIN_TS_GAP:
                return False
        picked.append(path)
        used_scenes[sk] += 1
        return True

    for _, path in scored:
        if try_add(path, enforce_scene_cap=True) and len(picked) >= n:
            return picked

    for _, path in scored:
        if path in picked:
            continue
        if try_add(path, enforce_scene_cap=False) and len(picked) >= n:
            break

    return picked[:n]


def draw_detections(img_bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    out = img_bgr.copy()
    for det in dets:
        x1, y1, x2, y2 = map(int, det["box"])
        color = COLOR_BGR.get(det["color"], COLOR_BGR["Unknown"])
        label = f'{det["color"]} {det["confidence"]:.2f}'
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(0, y1 - 6)
        cv2.rectangle(
            out,
            (x1, ty - th - baseline - 4),
            (x1 + tw + 4, ty + 2),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1 + 2, ty - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def resize_width(img_bgr: np.ndarray, width: int = FRAME_WIDTH) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w == width:
        return img_bgr
    new_h = int(round(h * (width / w)))
    return cv2.resize(img_bgr, (width, new_h), interpolation=cv2.INTER_AREA)


def main() -> None:
    with open(BOXES_PATH) as f:
        results = json.load(f)

    by_path: dict[str, list[dict]] = collections.defaultdict(list)
    for row in results:
        by_path[row["filepath"]].append(row)

    paths = select_paths(by_path, N_FRAMES)
    if not paths:
        raise SystemExit(f"No frames to render from {BOXES_PATH}")

    frames: list[Image.Image] = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"skip missing image: {path}")
            continue
        annotated = draw_detections(img, by_path[path])
        annotated = resize_width(annotated)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
        colors = sorted({d["color"] for d in by_path[path]})
        print(f"  + {Path(path).name}  colors={colors}  n={len(by_path[path])}")

    if not frames:
        raise SystemExit("No frames could be loaded")

    # Adaptive palette keeps the GIF small enough for GitHub README embeds
    paletted = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(
        OUT_PATH,
        save_all=True,
        append_images=paletted[1:],
        duration=DURATION_MS,
        loop=0,
        optimize=True,
    )
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT_PATH} ({len(frames)} frames, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
