# nuScenes traffic lights pipeline
PYTHON   ?= .venv/bin/python
PIP      ?= .venv/bin/pip
SRC      := traffic_lights/src

.PHONY: help setup venv install \
        extract detect export vis gif \
        pipeline all clean clean-outputs \
        docker-build docker-pipeline docker-vis docker-down docker-shell \
        docker-gpu-check

help:
	@echo "Targets:"
	@echo "  make setup      - venv + deps"
	@echo "  make extract    - CAM_FRONT paths → paths_for_samples.json"
	@echo "  make detect     - YOLO + HSV → boxes_dict.json"
	@echo "  make export     - FiftyOne format → boxes_dict_for_fiftyone.json"
	@echo "  make vis        - FiftyOne App (local; needs mongod / URI)"
	@echo "  make gif        - 10-frame preview GIF → assets/predictions_preview.gif"
	@echo "  make pipeline   - extract → detect → export (no UI)"
	@echo "  make all        - pipeline + vis"
	@echo "  make clean      - JSON + debug_unknown/"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - build image"
	@echo "  make docker-pipeline  - pipeline in container (GPU)"
	@echo "  make docker-vis       - Mongo + FiftyOne → http://localhost:5151"
	@echo "  make docker-down      - stop and remove compose containers"
	@echo "  make docker-shell     - shell in image (GPU)"
	@echo "  make docker-gpu-check - nvidia-smi in CUDA container"

# --- env ---

venv:
	python3.12 -m venv .venv

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

setup: install

# --- steps (order = file dependencies) ---

paths_for_samples.json: $(SRC)/extract_paths.py
	$(PYTHON) $(SRC)/extract_paths.py

boxes_dict.json: paths_for_samples.json $(SRC)/detect_and_classify.py
	$(PYTHON) $(SRC)/detect_and_classify.py

boxes_dict_for_fiftyone.json: boxes_dict.json $(SRC)/export_fiftyone.py
	$(PYTHON) $(SRC)/export_fiftyone.py

extract: paths_for_samples.json
detect:  boxes_dict.json
export:  boxes_dict_for_fiftyone.json

vis: boxes_dict_for_fiftyone.json
	$(PYTHON) $(SRC)/visualize.py

assets/predictions_preview.gif: boxes_dict.json $(SRC)/make_preview_gif.py
	$(PYTHON) $(SRC)/make_preview_gif.py

gif: assets/predictions_preview.gif

pipeline: export
all: vis

# --- cleanup ---

clean-outputs:
	rm -f paths_for_samples.json boxes_dict.json boxes_dict_for_fiftyone.json
	rm -rf debug_unknown

clean: clean-outputs

# --- docker (GPU detect + FiftyOne via compose Mongo) ---

docker-build:
	docker compose build

docker-pipeline:
	docker compose run --rm pipeline

docker-vis:
	docker compose up vis

docker-down:
	docker compose down

docker-shell:
	docker compose run --rm --entrypoint bash pipeline

docker-gpu-check:
	docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
