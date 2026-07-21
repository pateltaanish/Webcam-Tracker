# Model Licenses

Every model actually pinned and used in this project, with its exact
license. Updated whenever a new model is added (see docs/02_architecture.md
sec 3 for the comparison/selection reasoning behind each choice).

## YOLO11n (person detector) -- Stage 1.3

| | |
|---|---|
| Source | Ultralytics (`ultralytics` PyPI package, version `8.4.102`) |
| Weights file | `yolo11n.pt`, auto-downloaded from `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt` into `models/yolo11n.pt` (gitignored -- not committed to the repo) |
| Trained on | COCO (80 classes; we filter to the `person` class only, id 0) |
| License | **AGPL-3.0** |
| Commercial-use note | AGPL-3.0 requires that if this software is distributed or run as a network service for others, the complete corresponding source code must be made available under AGPL-3.0 terms. Ultralytics separately sells an Enterprise license for closed-source commercial use that isn't AGPL-compliant. **Current status: this project is personal/research development, not distributed -- AGPL-3.0 imposes no obligation yet.** Revisit this before any distribution, sale, or hosted/network use of the software. |
| Where used | `src/webcam_tracker/detection/detector.py` (`PersonDetector`) |

## Not yet added

Face detection/embedding (SCRFD + ArcFace) and person re-identification
(OSNet) are planned for Stage 2 -- see docs/02_architecture.md sec 3.3-3.4
for the comparison, and this file will get an entry for each once their
exact weight files are pinned (some InsightFace pretrained checkpoints carry
non-commercial research restrictions that must be checked per-file before
use, not assumed from the code license alone).
