# Gameplay Event Detection from Video

An end-to-end computer vision system for gameplay event detection, combining object detection, tracking, minimap analysis, OCR, and temporal event reasoning.

The project converts selected regions of gameplay video into structured, time-stamped events. It is an interpretable engineering prototype built around pretrained YOLO11n detections, lightweight IoU tracking, HUD OCR, and a color-and-shape minimap adapter. The reference configuration and reported results are specific to one documented PUBG Mobile clip; this repository does not claim general gameplay understanding or production-level cross-game performance.

## Contents

- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Current Limitations](#current-limitations)
- [Future Work](#future-work)
- [Data and Attribution](#data-and-attribution)

## Core Features

- Character and vehicle detection with pretrained YOLO11n
- Configurable semantic mapping: `person -> character`, `car/truck -> vehicle`
- Per-class bounding-box filters and semantic non-maximum suppression
- Class-aware, greedy IoU tracking with persistent internal IDs
- Temporal confirmation, disappearance detection, and event deduplication
- EasyOCR-based magazine-ammo recognition with multi-observation validation
- Yellow minimap-marker localization using HSV segmentation and contour geometry
- JSON and CSV output for lifecycle and ammo-state events
- Detection-level and event-level evaluation
- Annotated demo generation and representative failure analysis
- 31 unit tests covering the core deterministic behavior

## System Architecture

```text
Video
├── YOLO11n
│   └── semantic mapping → geometry filters → semantic NMS
│       └── class-aware IoU tracking → temporal lifecycle reasoning
│           └── character/vehicle events → JSON / CSV
├── HUD ROI
│   └── resize → EasyOCR → numeric parsing → temporal validation
│       └── ammo_changed events → JSON / CSV
└── Minimap ROI
    └── HSV mask → contours → area/circularity filters
        └── normalized ROI-relative marker coordinates → analysis / demo
```

The three branches address different visual problems: learned detection for scene objects, OCR for a fixed HUD number, and a transparent color-and-shape heuristic for one distinctive minimap marker.

Minimap observations are not currently serialized into the primary event stream. The main pipeline can run the analyzer when enabled, while the dedicated demo consumes its coordinates for visualization.

## Repository Structure

```text
Gameplay_Event_Detection/
├── main.py                         # Compatibility CLI entry point
├── config.example.json             # Reference pipeline configuration
├── requirements.txt                # Python dependencies
├── annotations.json                # Reviewed detection annotations
├── annotations.example.json        # Minimal annotation schema example
├── events_ground_truth.json        # Event-level ground truth
├── events_ground_truth.example.json
├── src/
│   ├── main.py                     # Video loop and orchestration
│   ├── config.py                   # Typed config loading and validation
│   ├── detector.py                 # YOLO, filtering, semantic NMS
│   ├── tracker.py                  # Class-aware greedy IoU tracker
│   ├── event_engine.py             # Temporal lifecycle state machine
│   ├── ocr.py                      # ROI OCR and value confirmation
│   ├── minimap.py                  # HSV/contour marker localization
│   ├── evaluator.py                # Detection/event matching metrics
│   ├── output.py                   # JSON/CSV writers
│   ├── visualizer.py               # General annotated-video writer
│   └── demo.py                     # Reference-clip demo renderer
└── tests/                           # Unit tests
```

Videos, model weights, generated outputs, caches, virtual environments, IDE settings, packaged submissions, and rendered demos are intentionally excluded from version control. Output directories are created automatically when needed.

## Installation

Python 3.10 or newer is recommended. The reference implementation was tested with Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Ultralytics downloads `yolo11n.pt` on first use when it is not cached. EasyOCR similarly downloads its models on first use. The reference configuration uses EasyOCR. The code also supports `pytesseract`, which requires a separately installed Tesseract executable.

## Quick Start

### 1. Obtain the reference video

Download the documented [Wikimedia Commons source](https://commons.wikimedia.org/wiki/File:PUBG_MOBILE_-_%C2%A1No_te_pierdas_todas_las_actualizaciones_del_modo_cl%C3%A1sico_aqu%C3%AD_mismo!.webm) and save it as `gameplay_classic.webm` in the repository root.

The CLI accepts any local video path, but the reference mappings, ROIs, time windows, annotations, and reported metrics are calibrated for this clip.

### 2. Run event detection

```bash
python -m src.main \
  --video gameplay_classic.webm \
  --config config.example.json
```

The compatibility entry point is equivalent:

```bash
python main.py --video gameplay_classic.webm --config config.example.json
```

Use `python -m src.main --help` for CLI options.

### 3. Generate the annotated demo

Run the main pipeline first so that the event file exists, then run:

```bash
python -m src.demo \
  --video gameplay_classic.webm \
  --config config.example.json \
  --events outputs/events.json \
  --output outputs/demo.mp4
```

The demo is intentionally tied to the validated character, vehicle, ammo, and minimap sections of the reference clip.

## Configuration

`config.example.json` is loaded into typed dataclasses and validated before processing.

| Section | Purpose | Reference settings |
|---|---|---|
| `detector` | Model, source classes, semantic mapping, filters, NMS | YOLO11n, confidence `0.25`, `imgsz=960`, NMS IoU `0.3` |
| `analysis_segments` | Sorted, non-overlapping windows | `37.0–42.5s`, `79.0–86.0s` |
| `tracker` | IoU association and track retention | IoU `0.1`, retention `30` frames |
| `temporal_smoothing` | Lifecycle confirmation/disappearance | confirm `5`, disappear `15` frames |
| `ocr` | ROI, sampling, scale, confidence, stability | EasyOCR, every `5` frames, `4x`, confidence `0.25`, stability `3` |
| `minimap` | ROI and HSV/shape constraints | disabled in the main reference config |
| `output` | Event and optional video paths | JSON, CSV, optional MP4 |
| `evaluation` | Matching thresholds | detection IoU `0.5`, event tolerance `0.5s` |

Normalized ROIs use `[x1, y1, x2, y2]` values in `[0, 1]`. Class filters use box area relative to the full frame and width/height aspect ratio.

Validation enforces `tracker.max_missing_frames >= temporal_smoothing.disappearance_frames`, keeping an identity alive long enough for the event engine to declare disappearance. The segment `purpose` field is descriptive and used in logs; it does not dynamically restrict detector classes.

## Output Format

The main pipeline creates parent directories and writes:

- `outputs/events.json`: structured event objects
- `outputs/events.csv`: the same event stream in tabular form
- `outputs/annotated.mp4`: only if `visualization_enabled` is `true`

Object lifecycle event:

```json
{
  "timestamp": 37.16379576703229,
  "frame_id": 1115,
  "event": "vehicle_detected",
  "object_id": 2,
  "object_type": "vehicle",
  "confidence": 0.5970599055290222,
  "bbox": [809.0297241210938, 534.09228515625, 1022.7211303710938, 655.2421875]
}
```

OCR-derived event:

```json
{
  "timestamp": 80.99374324115557,
  "frame_id": 2430,
  "event": "ammo_changed",
  "object_id": "ammo",
  "object_type": "ammo",
  "confidence": 0.9338584097218197,
  "bbox": null,
  "metadata": {
    "previous_value": 14,
    "current_value": 12,
    "source": "ocr",
    "confidence_available": true
  }
}
```

`object_id` is an internal tracker identity, not an in-game account ID. `ammo_changed` describes an observed HUD transition, not necessarily `shot_fired`: reloads, weapon changes, edits, and OCR errors can also change the display.

## Evaluation

The repository keeps the reviewed annotations and evaluation code; generated metrics and media are ignored by Git. Reproduce metrics using the pipeline and the functions in `src/evaluator.py` with the supplied annotations.

### Detection protocol

- 55 unique annotated frames
- 167 GT instances: 142 characters and 25 vehicles
- Same-frame, same-class matching
- Greedy one-to-one assignment by descending IoU
- Match threshold: IoU >= 0.5

Reference results:

| Class | GT | Pred. | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Character | 142 | 143 | 139 | 4 | 3 | 0.9720 | 0.9789 | 0.9754 |
| Vehicle | 25 | 24 | 24 | 0 | 1 | 1.0000 | 0.9600 | 0.9796 |
| Overall | 167 | 167 | 163 | 4 | 4 | 0.9760 | 0.9760 | 0.9760 |

### Event protocol

Predictions are matched one-to-one by event type and closest timestamp within `±0.5s`. Segment-initialization predictions are excluded using one deterministic, type-independent rule applied before matching.

Reference result: TP `4`, FP `4`, FN `4`, precision `0.5000`, recall `0.5000`, and F1 `0.5000`.

The gap between frame-level detection and event-level performance exposes downstream limitations: track fragmentation creates extra lifecycle events, while OCR confirmation can skip short-lived ammo values.

These results come from a small, targeted, prediction-assisted and manually reviewed validation set on one video. They are not broad benchmark or cross-game claims. The evaluator does not compute AP or mAP.

## Testing

```bash
python -m compileall -q src tests main.py
python -m unittest discover -v
```

The 31 tests cover configuration validation, semantic mapping and NMS, IoU tracking, temporal confirmation/disappearance, one-to-one evaluation, boundary exclusion, OCR smoothing, minimap normalization, output serialization, and invalid video metadata.

The tests use mocks and synthetic observations for deterministic unit behavior; they do not benchmark real YOLO accuracy.

## Current Limitations

- Mappings, ROIs, thresholds, and analysis windows are calibrated to one PUBG Mobile clip.
- YOLO11n is pretrained on natural-image categories and is not fine-tuned on gameplay footage.
- `character` does not imply enemy or teammate affiliation.
- The greedy IoU tracker has no motion model or appearance features and may fragment IDs during occlusion, overlap, abrupt camera motion, or box jitter.
- Frame-count temporal thresholds represent different durations at different source frame rates.
- Geometry filters can reject real objects; one documented vehicle FN came from an oversized `truck` prediction rejected by the area limit.
- Semantic NMS can retain a coarse high-confidence character box and suppress a tighter localization.
- OCR uses a fixed HUD ROI; subtitles, layout changes, small text, and rapid transitions reduce reliability.
- Minimap localization recognizes one configured yellow marker, returns only ROI-relative coordinates, and is not part of the JSON/CSV event stream.
- Event evaluation matches type and time but does not validate ammo transition metadata or object IDs.
- The evaluation set is small and should be treated as targeted validation.

## Future Work

- Fine-tune a gameplay-specific detector using independent train/validation/test splits.
- Evaluate thresholds on separate development and held-out clips.
- Add motion and appearance cues for longer occlusion and stronger identity continuity.
- Express temporal thresholds in seconds rather than frames.
- Add dynamic HUD localization and a HUD-specific digit model.
- Extend minimap detection across marker types and UI themes.
- Define minimap observation semantics before adding them to the event stream.
- Calibrate minimap positions to world coordinates only when scale, rotation, and map landmarks are available.
- Add value-aware event evaluation and metrics across multiple IoU thresholds.

## Data and Attribution

The reference evaluation uses:

- Title: *PUBG MOBILE - ¡No te pierdas todas las actualizaciones del modo clásico aquí mismo!*
- Author: PUBG MOBILE LATAM
- Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:PUBG_MOBILE_-_%C2%A1No_te_pierdas_todas_las_actualizaciones_del_modo_cl%C3%A1sico_aqu%C3%AD_mismo!.webm)
- Original publication: [YouTube](https://www.youtube.com/watch?v=qGawy7LcBfc)
- License: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)
- Expected SHA-1: `e9768bd3b90d342abfd945006eca2fea32994a6e`
- Expected size: 57,574,789 bytes

The video is not tracked in this repository. Download it from the attributed source and comply with its license.
