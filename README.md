# Gameplay Event Detection from Video

## Overview

This project is a modular gameplay video understanding pipeline that converts visual observations into structured, time-stamped events. It combines object detection, semantic class mapping, configurable filtering, multi-object tracking, and temporal lifecycle reasoning:

`video -> YOLO detection -> semantic mapping/filtering -> tracking -> temporal reasoning -> structured events`

Two complementary visual channels extend the object pipeline: a HUD region is processed with OCR to observe magazine-ammo state changes, while a minimap region is analyzed with color and shape constraints to localize a configured marker. The system is deliberately scoped as an interpretable engineering prototype; game-specific mappings, regions of interest, thresholds, and analysis windows remain explicit and configurable.

## Key Capabilities

- Character and vehicle detection with YOLO11n
- Configurable mapping from detector labels to gameplay semantics
- Per-class geometric filtering and semantic non-maximum suppression
- Cross-frame object tracking with persistent IDs
- Temporally smoothed lifecycle event generation and deduplication
- HUD magazine-ammo recognition with multi-observation validation
- Minimap marker localization in normalized ROI-relative coordinates
- Structured JSON and CSV output for lifecycle and ammo-state events
- Class-aware detection and time-tolerant event evaluation
- Representative failure-case analysis
- Annotated demo generation

## Setup

Tested with Python 3.13. A Python 3.10+ virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Ultralytics downloads `yolo11n.pt` on first use if it is not already available. EasyOCR similarly obtains its model on first use; normal inference does not install packages at runtime. To avoid model downloads during review, run once while online or provide the cached weights locally.

EasyOCR is the backend used for the reference results. `pytesseract` remains an optional alternative backend supported by the code and additionally requires a system-installed Tesseract executable. The reference configuration, demo, and evaluation do not require a Tesseract executable.

## Usage

The gameplay source video is intentionally not bundled to keep the repository and packaged artifacts manageable. Download the exact [Wikimedia Commons source documented below](https://commons.wikimedia.org/wiki/File:PUBG_MOBILE_-_%C2%A1No_te_pierdas_todas_las_actualizaciones_del_modo_cl%C3%A1sico_aqu%C3%AD_mismo!.webm), save it as `gameplay_classic.webm` in the project root, then run:

```bash
python -m src.main \
  --video gameplay_classic.webm \
  --config config.example.json
```

The pipeline does not require this fixed filename: `--video` may point to another local video. The included semantic mappings, ROIs, analysis windows, artifacts, and reported validation results are specific to the documented clip.

Generate the annotated demo:

```bash
python -m src.demo --video gameplay_classic.webm --config config.example.json \
  --events outputs/events.json --output outputs/demo.mp4
```

The packaged demo is additionally encoded with H.264 (libx264, CRF 25) to reduce repository/package size; this encoding step does not alter the underlying annotations or event data.

Use `python -m src.main --help` and `python -m src.demo --help` for all CLI options.

## Outputs

- `outputs/events.json`: raw pipeline events, including segment-initialization events
- `outputs/events.csv`: the same event stream in tabular form
- `outputs/demo.mp4`: 31.13-second, 1920x1080 annotated demo
- `outputs/evaluation/`: detection metrics, event metrics, predictions, and the excluded-event audit
- `outputs/analysis/`: representative detection, OCR, and minimap examples

These examples are copied from the included reference outputs:

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

```json
{
  "timestamp": 80.99374324115557,
  "frame_id": 2430,
  "event": "ammo_changed",
  "object_id": "ammo",
  "object_type": "ammo",
  "confidence": 0.9338584097218197,
  "bbox": null,
  "metadata": {"previous_value": 14, "current_value": 12, "source": "ocr", "confidence_available": true}
}
```

## System Architecture

```text
Video
├── YOLO -> Semantic Mapping/Filtering -> Tracking -> Lifecycle Events -> JSON/CSV
├── HUD ROI -> OCR -> Temporal Validation -> Ammo State Events ----------> JSON/CSV
└── Minimap ROI -> HSV/Shape Analysis -> Marker Coordinates -------------> Analysis/Demo
```

- `detector.py`: detector interface and Ultralytics YOLO implementation
- `tracker.py`: lightweight, class-aware IoU tracker
- `event_engine.py`: confirmation, disappearance, lifecycle state, and deduplication
- `ocr.py`: configurable ROI OCR and multi-observation value confirmation
- `minimap.py`: configurable ROI and HSV/shape marker localization
- `evaluator.py`: one-to-one detection/event matching and boundary partitioning
- `visualizer.py`: general annotated-video writer
- `demo.py`: concise, video-specific visualization of the evaluated sections
- `config.py`: typed JSON loading and validation

Configuration validation requires `max_missing_frames >= disappearance_frames`. This keeps a tracker identity alive long enough for the EventEngine to emit its disappearance event before the track is retired.

## Video and Evaluation Setup

Evaluation was conducted on a representative PUBG Mobile gameplay clip sourced from Wikimedia Commons under CC BY 3.0. The original assessment materials referenced a gameplay video but did not include one, so this public clip provides a reproducible evaluation source without being presented as an official video.

- Exact title: *PUBG MOBILE - ¡No te pierdas todas las actualizaciones del modo clásico aquí mismo!*
- Author/attribution: PUBG MOBILE LATAM
- Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:PUBG_MOBILE_-_%C2%A1No_te_pierdas_todas_las_actualizaciones_del_modo_cl%C3%A1sico_aqu%C3%AD_mismo!.webm) (originally published on [YouTube](https://www.youtube.com/watch?v=qGawy7LcBfc))
- License: [Creative Commons Attribution 3.0 Unported](https://creativecommons.org/licenses/by/3.0/)
- Local file: renamed to `gameplay_classic.webm`; otherwise unmodified. Its SHA-1 (`e9768bd3b90d342abfd945006eca2fea32994a6e`) and 57,574,789-byte size match the Commons source.
- Metadata: 1920x1080, 30.0023 FPS, 8,290 decoded frames, approximately 276.31 seconds

The source is a montage/tutorial compilation rather than a continuous match. Subsystem evaluation therefore uses continuous gameplay windows—vehicle `37.0-42.5s` and character/ammo `79.0-86.0s`—to avoid interpreting editing cuts as temporal events. Minimap localization is evaluated separately on `114.0-121.5s`.

## Object Detection

YOLO11n runs at `imgsz=960` with a confidence threshold of `0.25`.

### Semantic Mapping

The configurable mapping is applied before tracking:

```text
YOLO person -> character
YOLO car    -> vehicle
YOLO truck  -> vehicle
```

`character` is used instead of `enemy` because a general detector cannot reliably infer teammate/enemy affiliation; visible humans also include the controlled avatar and training targets. `car` and `truck` are merged because YOLO alternated those COCO labels for the same UAZ across nearby frames. `motorcycle` and `bus` are excluded because inspection did not establish them as reliable real gameplay objects in the selected clip.

### Filtering and Semantic NMS

Configurable bounding-box area and aspect-ratio filters reject implausible detections before tracking. Semantic NMS then suppresses overlapping boxes after source classes have been mapped, preventing `car` and `truck` predictions for the same vehicle from creating separate semantic identities. The filters use normalized geometric ratios rather than coordinates fitted to one object.

## Multi-Object Tracking

The reference pipeline uses a class-aware IoU tracker with an IoU threshold of `0.1` and a 30-frame missing-track retention window. A track is confirmed after 5 consecutive observations. Once confirmed, it must remain absent for 15 frames before a disappearance event is emitted.

On the selected validation windows, the IoU tracker produced one vehicle ID and eight character IDs. An Ultralytics ByteTrack comparison produced two vehicle IDs and 15 character IDs and introduced an additional `lap` dependency. The IoU tracker was retained because it was more stable on these specific windows while remaining transparent and dependency-light; this is not a claim that IoU tracking is universally superior.

Tracker and EventEngine state reset at each configured analysis-window boundary. This prevents a montage transition from manufacturing a disappearance event. Initialization events remain in the raw output because they accurately describe pipeline state, but the evaluation protocol treats them explicitly as described below.

## Gameplay Event Recognition

Object events are derived from confirmed track lifecycles rather than emitted on every frame:

```text
new track -> consecutive observations -> <class>_detected
temporary miss -> retain lifecycle state -> no event
persistent miss -> disappearance window reached -> <class>_disappeared
analysis boundary -> reset state -> no manufactured disappearance
```

The reference event vocabulary is:

- `character_detected`
- `character_disappeared`
- `vehicle_detected`
- `ammo_changed`

`ammo_changed` means only that the HUD-observed magazine value changed. It is not equivalent to `shot_fired`: reloads, weapon switches, or editing transitions can also change the display.

## Evaluation

### Evaluation Protocol

Detection evaluation uses IoU >= 0.5 with greedy, one-to-one, class-aware matching. The annotation set contains 55 unique frames and 167 GT instances. Initial boxes were prediction-assisted, followed by manual review and correction of every sampled frame and bounding box; detector misses were added manually. The result is a targeted, manually reviewed validation set rather than a blind test set or broad gameplay benchmark.

### Object Detection Performance

| Class | Annotated frames | GT | Pred. | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Character | 32 | 142 | 143 | 139 | 4 | 3 | 0.9720 | 0.9789 | 0.9754 |
| Vehicle | 25 | 25 | 24 | 24 | 0 | 1 | 1.0000 | 0.9600 | 0.9796 |
| Overall | 55 unique | 167 | 167 | 163 | 4 | 4 | 0.9760 | 0.9760 | 0.9760 |

Two annotated frames contain both character and vehicle instances, so the per-class annotated-frame counts sum to 57 while the evaluation set contains 55 unique frames. Representative reviewed errors are available in `outputs/analysis/failures/`.

### Event Recognition Performance

Event ground truth was manually timestamped only inside continuous gameplay windows. Matching requires the same event type, a timestamp difference of at most +/-0.5 seconds, and one-to-one pairing.

The scoring scope applies one deterministic initialization rule to every event type before matching: predictions in `[segment_start, segment_start + min_confirmation_frames / source_fps]` are excluded. Here the window is `5 / 30.0023 = 0.166654s`. GT montage-cut boundary events and predicted initialization events are retained in `excluded_boundary_events.json`; they are auditable but count as neither TP nor FP/FN. Six predicted initialization events and one GT montage-boundary event were excluded. No unmatched event is excluded based on its effect on a metric.

| Event | Scored pred. | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `character_detected` | 3 | 1 | 1 | 2 | 0 | 0.3333 | 1.0000 | 0.5000 |
| `character_disappeared` | 2 | 1 | 1 | 1 | 0 | 0.5000 | 1.0000 | 0.6667 |
| `vehicle_detected` | 0 | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| `ammo_changed` | 3 | 6 | 2 | 1 | 4 | 0.6667 | 0.3333 | 0.4444 |
| Overall | 8 | 8 | 4 | 4 | 4 | 0.5000 | 0.5000 | 0.5000 |

The overall event result is TP=4, FP=4, FN=4, with precision, recall, and F1 all equal to 0.5000. This lower event-level score is retained without retrospective threshold tuning and exposes lifecycle fragmentation and OCR recall as areas for improvement.

When both the scored ground-truth count and prediction count are zero, precision, recall, and F1 are not meaningful. The evaluator serializes zero for zero-denominator cases, while this README presents them as N/A for clarity. The `vehicle_detected` row is therefore not interpreted as a performance failure.

## HUD Ammo Recognition

The `79.0-86.0s` window contains a fixed, visible magazine counter and real value changes. Its normalized ROI is `[0.35833, 0.94907, 0.37240, 0.98889]` (pixels `688:715, 1025:1068` at 1920x1080). The selected pipeline uses EasyOCR, a tight crop, 4x bicubic scaling, a digit allowlist, sampling every five frames, confidence >= 0.25, and three matching observations before state confirmation.

A small crop-level validation used 29 manually inspected representative crops: 24 were correctly parsed (82.8%). This is an ROI feasibility check, not an independent OCR benchmark or a claim of generalization. During the sequential run, 43 observations were sampled, 36 were parseable, and 34 met the confidence threshold. Temporal validation emitted three stable `ammo_changed` events. This smoothing intentionally trades recall and latency for stability: intermediate values can be skipped when the HUD changes faster than the three-observation window.

`pytesseract` is supported as an optional alternative and requires a separately installed Tesseract executable. The reference demo and reported evaluation use EasyOCR and do not require Tesseract.

## Minimap Marker Localization

The minimap adapter detects one configured yellow waypoint using HSV segmentation, contour-area bounds, and a circularity constraint. In the selected `114.0-121.5s` validation window, the marker was detected in 76/76 sampled frames, with no false positives across 178 sampled negative frames. These figures describe only the selected validation samples and do not represent general minimap accuracy.

The heuristic is yellow-waypoint-specific and does not perform general map understanding. Output coordinates are normalized relative to the on-screen minimap ROI and are not global world coordinates. Minimap localization is currently exposed as an independently evaluated analysis and visualization channel: the demo visualizes its marker positions, but these observations are not serialized into the primary JSON/CSV gameplay event stream. The current structured event output contains track-lifecycle events and OCR-derived `ammo_changed` events. If the ROI is absent or no qualifying contour exists, the analyzer returns no observation rather than inventing a disappearance event.

## Failure Analysis and Limitations

- Character predictions can form nested or coarse boxes around the controlled avatar. Semantic NMS may retain the coarse box and suppress a tighter localization, creating paired FP/FN at IoU 0.5.
- One visible UAZ became an FN because YOLO produced only an oversized, scope-overlapping `truck` box, which the area filter rejected.
- Spanish subtitles overlap the lower HUD and interfere with ammo OCR.
- OCR smoothing skips short-lived intermediate ammo values in exchange for fewer noisy events.
- Montage cuts invalidate ordinary lifecycle continuity; segment gating limits analysis to validated continuous windows rather than claiming full-video coverage.
- `character` includes any human-like detected subject; team affiliation and character role are not inferred.
- The minimap heuristic is specific to the selected yellow waypoint, and minimap-relative coordinates are not world localization.
- Pretrained YOLO was trained on natural-image categories rather than PUBG Mobile, so domain shift, small or distant subjects, occlusion, and motion blur remain risks.
- The current evaluation set is intentionally limited in scale and should be interpreted as targeted system validation rather than a broad gameplay benchmark.

Representative examples include `outputs/analysis/failures/character_fp_1_frame_2370.jpg`, `character_fn_1_frame_2370.jpg`, and `vehicle_fn_1_frame_1123.jpg`.

## Design Decisions and Trade-offs

- The vehicle area filter favors precision over recall; it removes scope/HUD false positives but rejected one real UAZ observation.
- Three-observation OCR confirmation favors stable changes over recall and latency.
- The simple IoU tracker has fewer dependencies and performed better on the evaluated windows, but appearance-aware tracking could be stronger under longer occlusion or abrupt camera motion.
- Segment gating preserves defensible lifecycle semantics for a montage at the cost of full-video coverage.
- Explicit, configurable adapters keep the pipeline easy to inspect, but game-specific ROIs and semantics must be recalibrated for a different title or HUD layout.

## Performance

Performance was measured on an AMD Ryzen 9 7945HX (16 cores / 32 logical CPUs) under WSL2 using CPU inference only; GPU access was unavailable. With YOLO11n, `imgsz=960`, EasyOCR enabled, and the evaluated configuration:

- source: 1920x1080 at 30.0023 FPS
- decoded/read frames through the last selected window: 2,581
- detector-analyzed frames: 377 (166 vehicle-window + 211 character-window)
- wall time: 38.52 seconds
- approximate end-to-end throughput: 9.79 analyzed frames/second
- peak resident memory reported by `/usr/bin/time`: 2,236,916 KB

This measurement includes video decoding, detection, tracking, event logic, OCR, and JSON/CSV writing for the selected analysis windows. It excludes one-time dependency/model downloads and demo rendering.

## Testing

```bash
python -m compileall -q src tests main.py
python -m unittest discover -v
```

The suite contains 31 tests covering configuration validation, semantic mapping/filtering, IoU and tracking lifecycle, temporal confirmation/disappearance/deduplication, one-to-one evaluation, deterministic initialization exclusion, OCR smoothing, minimap normalization, serialization, and invalid video metadata.

## Future Work

Practical extensions include a domain-adapted character/vehicle detector, held-out tuning and evaluation splits, appearance-aware tracking for long occlusion, a HUD-specific digit model, broader minimap-marker models, and calibration from minimap-relative positions to global world coordinates.
