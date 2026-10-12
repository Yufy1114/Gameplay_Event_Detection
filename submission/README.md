# Gameplay Event Detection from Video

## Overview

This project turns gameplay video into structured events through a readable pipeline:

`video -> YOLO detection -> semantic mapping/filtering -> tracking -> temporal reasoning -> JSON/CSV`

It is a small, reproducible take-home implementation rather than a claim of production-level game understanding. The checked-in configuration is adapted to the supplied representative PUBG Mobile clip and keeps every game-specific mapping, ROI, threshold, and analysis segment explicit.

## Features

Core:

- `character` detection and cross-frame tracking
- `vehicle` detection and cross-frame tracking
- `character_detected`, `character_disappeared`, and `vehicle_detected` lifecycle events
- temporal confirmation, disappearance smoothing, and event deduplication
- JSON and CSV event output
- a short annotated submission demo

Bonus validation:

- EasyOCR magazine-ammo observations and stable `ammo_changed` events
- HSV-based yellow minimap-waypoint localization
- class-aware one-to-one detection evaluation
- type-aware one-to-one event evaluation

## Setup

Tested with Python 3.13. A Python 3.10+ virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Ultralytics downloads `yolo11n.pt` on first use if it is not already available. EasyOCR similarly obtains its model on first use; normal inference does not install packages at runtime. To avoid model downloads during review, run once while online or provide the cached weights locally.

EasyOCR is the backend used for the submitted results. `pytesseract` remains an optional alternative backend supported by the code and additionally requires a system-installed Tesseract executable. The submitted configuration, demo, and evaluation do not require a Tesseract executable.

## Run

The gameplay source video is intentionally not bundled in the submission package to keep its size manageable. Download the exact [Wikimedia Commons source documented below](https://commons.wikimedia.org/wiki/File:PUBG_MOBILE_-_%C2%A1No_te_pierdas_todas_las_actualizaciones_del_modo_cl%C3%A1sico_aqu%C3%AD_mismo!.webm), save it as `gameplay_classic.webm` in the project root, then run:

```bash
python -m src.main \
  --video gameplay_classic.webm \
  --config config.example.json
```

The pipeline does not require this fixed filename: `--video` may point to another local video. However, the submitted semantic mappings, ROIs, analysis segments, artifacts, and reported validation results are specific to the documented clip.

Generate the short submission demo:

```bash
python -m src.demo --video gameplay_classic.webm --config config.example.json \
  --events outputs/events.json --output outputs/demo.mp4
```

Use `python -m src.main --help` and `python -m src.demo --help` for all CLI options.

## Output

- `outputs/events.json`: raw pipeline events, including segment-initialization events
- `outputs/events.csv`: the same event stream in tabular form
- `outputs/demo.mp4`: 31.13-second, 1920x1080 annotated demo
- `outputs/evaluation/`: detection metrics, event metrics, predictions, and excluded-event audit

These examples are copied from the actual submitted `outputs/events.json`:

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

The `ammo_changed` event means only that the HUD-observed magazine value changed. It must not be interpreted as `shot_fired`; reloads, weapon switches, or edits can also change that display.

## Architecture

- `detector.py`: detector interface plus Ultralytics YOLO implementation
- semantic mapping/filtering: applied inside detector post-processing, before tracking
- `tracker.py`: lightweight, class-aware IoU tracker
- `event_engine.py`: confirmation, disappearance, lifecycle state, and deduplication
- `ocr.py`: configurable ROI OCR and multi-observation value confirmation
- `minimap.py`: configurable ROI plus HSV/shape marker localization
- `evaluator.py`: one-to-one detection/event matching and boundary partitioning
- `visualizer.py`: general annotated-video writer
- `demo.py`: concise, video-specific presentation of the validated sections
- `config.py`: typed JSON loading and validation

The tracker is intentionally retained until the EventEngine's disappearance window has elapsed. Configuration validation requires `max_missing_frames >= disappearance_frames`, preventing the tracker from deleting an ID before its lifecycle can emit a disappearance event.

## Video and data

The assessment referenced a gameplay video, but no official video was included with the provided materials. To complete and evaluate the pipeline within the submission window, a representative PUBG Mobile gameplay clip was used. This is not presented as a Razer-provided video.

- Exact title: *PUBG MOBILE - ¡No te pierdas todas las actualizaciones del modo clásico aquí mismo!*
- Author/attribution: PUBG MOBILE LATAM
- Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:PUBG_MOBILE_-_%C2%A1No_te_pierdas_todas_las_actualizaciones_del_modo_cl%C3%A1sico_aqu%C3%AD_mismo!.webm) (originally published on [YouTube](https://www.youtube.com/watch?v=qGawy7LcBfc))
- License: [Creative Commons Attribution 3.0 Unported](https://creativecommons.org/licenses/by/3.0/)
- Local file: renamed to `gameplay_classic.webm`; otherwise unmodified. Its SHA-1 (`e9768bd3b90d342abfd945006eca2fea32994a6e`) and 57,574,789-byte size match the Commons source.
- Metadata: 1920x1080, 30.0023 FPS, 8,290 decoded frames, approximately 276.31 seconds

The source is a montage/tutorial compilation rather than one continuous match. Evaluation therefore uses continuous gameplay windows: vehicle `37.0-42.5s` and character/ammo `79.0-86.0s`. This prevents montage cuts from being interpreted as in-game object lifecycles. The minimap bonus is validated separately on `114.0-121.5s`.

## Object types

The configurable semantic mapping is:

```text
YOLO person -> character
YOLO car    -> vehicle
YOLO truck  -> vehicle
```

`character` is used instead of `enemy`: a general detector cannot reliably infer teammate/enemy affiliation, and the visible humans include the controlled avatar and training targets. `car` and `truck` are merged before tracking because YOLO alternated those COCO labels for the same UAZ. `motorcycle` and `bus` were not included because inspection did not establish them as reliable real gameplay objects in the selected clip.

YOLO11n runs at `imgsz=960`, confidence `0.25`. Configurable area/aspect filters and semantic NMS reject obvious invalid boxes. They are deliberately generic ratios, not coordinates fitted to one object.

## Tracking and event logic

The selected IoU tracker uses an IoU threshold of `0.1` and retains missing tracks for 30 frames. A track is confirmed after 5 observations; a confirmed track must be absent for 15 frames before one `<class>_disappeared` event is emitted. A confirmed lifecycle emits `<class>_detected` only once.

On the selected segments, the IoU tracker produced one vehicle ID and eight character IDs. An Ultralytics ByteTrack comparison produced two vehicle IDs and 15 character IDs and introduced an additional `lap` dependency. IoU therefore performed better for these specific validation segments and was retained; this is not a claim that IoU tracking is universally superior.

Segment starts reset tracker and EventEngine state. Initialization detections remain in raw output because they accurately describe pipeline state, but are handled explicitly by the event-evaluation protocol below.

## Small-scale manually reviewed validation

### Detection validation

Evaluation uses IoU >= 0.5 and greedy one-to-one, class-aware matching. The annotation set contains 55 unique frames and 167 GT instances. Initial boxes were prediction-assisted to fit take-home time constraints, but every frame and bounding box was manually reviewed/corrected, and detector misses were manually added. This is not a blind test set.

| Class | Annotated frames | GT | Pred. | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Character | 32 | 142 | 143 | 139 | 4 | 3 | 0.9720 | 0.9789 | 0.9754 |
| Vehicle | 25 | 25 | 24 | 24 | 0 | 1 | 1.0000 | 0.9600 | 0.9796 |
| Overall | 55 unique | 167 | 167 | 163 | 4 | 4 | 0.9760 | 0.9760 | 0.9760 |

The selected samples are a small, manually reviewed validation set, not a large-scale benchmark. Representative reviewed errors are in `outputs/analysis/failures/`.

### Event validation

Event GT was manually timestamped only inside continuous gameplay. Matching requires the same event type, a timestamp difference of at most +/-0.5 seconds, and one-to-one assignment.

The scoring scope applies one deterministic initialization rule to every event type *before* matching: predictions in `[segment_start, segment_start + min_confirmation_frames / source_fps]` are excluded. Here the window is `5 / 30.0023 = 0.166654s`. GT montage-cut boundary events and predicted initialization events are retained in `excluded_boundary_events.json`; they are auditable but count as neither TP nor FP/FN. Six predicted initialization events and one GT montage-boundary event were excluded. No unmatched event is excluded merely because it hurts a metric.

| Event | Scored pred. | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `character_detected` | 3 | 1 | 1 | 2 | 0 | 0.3333 | 1.0000 | 0.5000 |
| `character_disappeared` | 2 | 1 | 1 | 1 | 0 | 0.5000 | 1.0000 | 0.6667 |
| `vehicle_detected` | 0 | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| `ammo_changed` | 3 | 6 | 2 | 1 | 4 | 0.6667 | 0.3333 | 0.4444 |
| Overall | 8 | 8 | 4 | 4 | 4 | 0.5000 | 0.5000 | 0.5000 |

This is small-scale event validation, not a benchmark. The weaker event score is reported without retuning the frozen detector, tracker, or lifecycle thresholds.

When both the scored ground-truth count and prediction count are zero, precision, recall, and F1 are not meaningful. The evaluator serializes zero for zero-denominator cases, while this README presents them as N/A for clarity. This row is therefore not interpreted as a `vehicle_detected` performance failure. Overall event precision, recall, and F1 remain 0.5000.

## OCR bonus

The selected `79.0-86.0s` segment has a fixed, visible magazine counter and real value changes. Its normalized ROI is `[0.35833, 0.94907, 0.37240, 0.98889]` (pixels `688:715, 1025:1068` at 1920x1080). The chosen pipeline is EasyOCR, tight crop, 4x bicubic scaling, digit allowlist, sampling every five frames, confidence >= 0.25, and three matching observations before confirmation. `pytesseract` is supported only as an optional alternative and requires a separately installed Tesseract executable; neither the submitted demo nor the reported evaluation uses it.

The small crop-level OCR validation used 29 manually inspected representative crops: 24 were correctly parsed (82.8%). This is an ROI feasibility check, not an independent OCR benchmark. During the sequential run, 43 observations were sampled, 36 were parseable, and 34 met the confidence threshold. Temporal validation emitted three stable `ammo_changed` events. It intentionally trades recall for stability: intermediate values can be skipped when the HUD changes faster than the three-observation window.

## Minimap bonus

The minimap adapter detects one configured yellow waypoint using HSV segmentation, contour area, and circularity constraints. In the selected `114.0-121.5s` validation segment, the marker was detected in 76/76 sampled frames, with no false positives across 178 sampled negative frames. This describes only the selected validation samples; it is not “100% minimap accuracy.”

The heuristic is yellow-waypoint-specific and does not understand the map. Output coordinates are normalized relative to the on-screen minimap ROI and do not represent global world coordinates. If the ROI is absent or no qualifying contour exists, the analyzer returns no observation rather than inventing a disappearance event.

## Failure cases and limitations

- Character predictions can form nested/coarse boxes around the controlled avatar. Semantic NMS may retain the coarse box and suppress a tighter localization, creating paired FP/FN at IoU 0.5.
- One visible UAZ became an FN because YOLO produced only an oversized scope-overlapping `truck` box, which the area filter rejected.
- Spanish subtitles overlap the lower HUD and interfere with ammo OCR.
- OCR smoothing skips short-lived intermediate ammo values in exchange for fewer noisy events.
- Montage cuts invalidate ordinary lifecycle continuity; segment gating limits analysis to validated continuous windows rather than claiming full-video coverage.
- `character` includes any human-like detected subject; team affiliation and character role are not inferred.
- The minimap heuristic is specific to the selected yellow waypoint, and minimap-relative coordinates are not world localization.
- Pretrained YOLO was trained on natural-image categories rather than PUBG Mobile, so domain shift, small/distant subjects, occlusion, and motion blur remain risks.
- The manually reviewed detection and event datasets are intentionally small because this is a take-home validation.

See `outputs/analysis/failures/character_fp_1_frame_2370.jpg`, `character_fn_1_frame_2370.jpg`, and `vehicle_fn_1_frame_1123.jpg` for representative cases.

## Trade-offs

- The vehicle area filter favors precision over recall; it removes scope/HUD false positives but rejected one real UAZ observation.
- Three-observation OCR confirmation favors stable changes over recall and latency.
- The simple IoU tracker has fewer dependencies and performed better on these segments, but appearance-aware tracking could be stronger under longer occlusion or abrupt camera motion.
- Segment gating produces defensible lifecycle semantics for a montage but does not claim continuous full-video coverage.

## Performance

Measured on an AMD Ryzen 9 7945HX (16 cores / 32 logical CPUs) under WSL2, CPU inference only; GPU access was unavailable. With YOLO11n, `imgsz=960`, EasyOCR enabled, and the submitted configuration:

- source: 1920x1080 at 30.0023 FPS
- decoded/read frames through the last selected segment: 2,581
- detector-analyzed frames: 377 (166 vehicle-window + 211 character-window)
- wall time: 38.52 seconds
- approximate end-to-end throughput: 9.79 analyzed frames/second
- peak resident memory reported by `/usr/bin/time`: 2,236,916 KB

This measurement includes video decoding, detection, tracking, event logic, OCR, and JSON/CSV writing for the selected analysis segments. It excludes one-time dependency/model downloads and demo rendering.

## Tests

```bash
python -m compileall -q src tests main.py
python -m unittest discover -v
```

The final suite contains 31 tests covering configuration validation, semantic mapping/filtering, IoU and tracking lifecycle, temporal confirmation/disappearance/deduplication, one-to-one evaluation, deterministic initialization exclusion, OCR smoothing, minimap normalization, serialization, and invalid video metadata.

## Future improvements

With more representative labeled gameplay, the next practical steps would be a domain-adapted character/vehicle detector, held-out tuning and evaluation splits, appearance-aware tracking for long occlusion, a HUD-specific digit model, and minimap calibration to world coordinates. Those are intentionally outside this take-home baseline.
