# Razer Computer Vision Internship — Interview Preparation

> Based strictly on the current repository implementation. English answers are intended for spoken interview practice; Chinese sections explain the underlying concepts.

## Table of Contents

1. [15-Minute Last-Minute Revision](#1-15-minute-last-minute-revision)
2. [Top 20 Most Likely Questions](#2-top-20-most-likely-questions)
3. [2-Minute Project Walkthrough](#3-2-minute-project-walkthrough)
4. [30-Second Project Summary](#4-30-second-project-summary)
5. [10 Things I Absolutely Must Know](#5-10-things-i-absolutely-must-know)
6. [Key Equations](#6-key-equations)
7. [Project Weaknesses / Red Flags](#7-project-weaknesses--red-flags)
8. [Full Detailed Interview Question Bank](#8-full-detailed-interview-question-bank)
9. [Live Modification Scenarios](#9-live-modification-scenarios)
10. [CV Fundamentals](#10-cv-fundamentals-derived-from-this-project)
11. [Rapid-Fire Questions](#11-rapid-fire-questions)
12. [Strongest Technical Decisions](#12-five-strongest-technical-decisions)

---

# 1. 15-Minute Last-Minute Revision

## Minutes 0–3: Architecture

Memorize:

```text
Video → YOLO → semantic mapping → geometry filtering
→ semantic NMS → IoU tracker → temporal engine → JSON/CSV

HUD → crop → 4× upscale → EasyOCR → confidence
→ 3 stable observations → ammo_changed

Minimap → ROI → HSV mask → contours → area/circularity
→ ROI-relative centroid
```

## Minutes 3–5: Exact values

- YOLO11n, `imgsz=960`
- confidence `0.25`
- semantic NMS IoU `0.3`
- tracker IoU `0.1`
- max missing `30`
- confirm `5`
- disappear `15`
- OCR every `5` frames
- OCR upscale `4×`
- OCR confidence `0.25`
- OCR stable observations `3`
- detection evaluation IoU `0.5`
- event time tolerance `±0.5s`

## Minutes 5–7: Metrics

- Detection: 163 TP / 4 FP / 4 FN / F1 0.9760
- Character: F1 0.9754
- Vehicle: F1 0.9796
- Event: 4 TP / 4 FP / 4 FN / F1 0.5
- Ammo event F1: 0.4444

## Minutes 7–9: Failure cases

- Coarse/nested avatar boxes.
- Semantic NMS sometimes keeps coarse box.
- Paired FP/FN at IoU 0.5.
- Oversized truck/scope prediction rejected, causing vehicle FN.
- Subtitle interference with OCR.
- Temporal OCR skips `13`, `11`, `9`.

## Minutes 9–11: Defensible limitations

- One clip.
- Small targeted dataset.
- Prediction-assisted annotation.
- Hand-configured ROIs and thresholds.
- IoU-only tracker.
- Frame-based temporal thresholds.
- Event evaluator ignores ammo values.
- Minimap not serialized.

## Minutes 11–13: Key distinctions

- confidence threshold ≠ NMS IoU ≠ tracker IoU ≠ evaluation IoU.
- detected event time ≠ first visible frame.
- disappearance confirmation time ≠ exact physical exit time.
- character ≠ enemy.
- ammo change ≠ shot.
- minimap-relative coordinate ≠ world coordinate.
- frame detection F1 ≠ event F1.

## Minutes 13–15: Practice aloud

Answer these without notes:

1. Walk me through the architecture.
2. Why is event F1 lower?
3. Why IoU tracker instead of ByteTrack?
4. Explain initialization exclusion honestly.
5. What would you change for production?

The strongest interview posture is: **be exact about what you built, candid about where it fails, and systematic about how you would validate the next improvement.**

---

# 2. Top 20 Most Likely Questions

| Rank | Probability | Question |
|---:|---|---|
| 1 | VERY HIGH | Walk me through the complete pipeline from frame to event. |
| 2 | VERY HIGH | How exactly does a detection become `character_detected` or `character_disappeared`? |
| 3 | VERY HIGH | Why is detection F1 0.976 but event F1 only 0.5? |
| 4 | VERY HIGH | How does your IoU tracker associate detections? |
| 5 | VERY HIGH | Why did you use a simple IoU tracker instead of ByteTrack/DeepSORT? |
| 6 | VERY HIGH | Explain confidence, NMS IoU, tracking IoU and evaluation IoU. |
| 7 | VERY HIGH | Why did you exclude initialization events? Is that fair? |
| 8 | VERY HIGH | What are the limitations of your small evaluation set? |
| 9 | VERY HIGH | Why `character`, not `enemy`? |
| 10 | HIGH | What is semantic NMS, and why map car/truck before it? |
| 11 | HIGH | What is the difference between `disappearance_frames` and `max_missing_frames`? |
| 12 | HIGH | Explain the ammo OCR pipeline and why it misses intermediate values. |
| 13 | HIGH | Explain the character FP/FN caused by coarse and nested boxes. |
| 14 | HIGH | Explain the vehicle FN caused by the geometry filter. |
| 15 | HIGH | What happens when FPS or resolution changes? |
| 16 | HIGH | Why does the project use configured analysis segments? |
| 17 | HIGH | What exactly does the minimap module return? |
| 18 | HIGH | What parts are reusable and what parts are clip-specific? |
| 19 | MEDIUM | How would you redesign the pipeline for production/live video? |
| 20 | MEDIUM | Why not train a custom detector or digit model? |

---

# 3. 2-Minute Project Walkthrough

> This project converts gameplay video into structured, time-stamped events. I designed it as three visual channels.
>
> The main object channel reads frames with OpenCV and runs YOLO11n at image size 960 with confidence threshold 0.25. I use the pretrained classes person, car and truck, map person to character and merge car and truck into vehicle. I then apply normalized geometry filters and semantic NMS after mapping, so overlapping car and truck predictions for the same UAZ do not create duplicate vehicle identities.
>
> The remaining detections go into a class-aware IoU tracker. It greedily associates current boxes with existing same-class tracks using IoU threshold 0.1. The event engine then performs temporal reasoning. A track must appear in five consecutive frames before a detected event is emitted. A confirmed track must be missing for fifteen frames before a disappeared event is emitted. The tracker retains missing IDs for up to thirty frames, and configuration validation ensures this is long enough for disappearance reasoning.
>
> The HUD channel crops a normalized magazine-ammo ROI, upscales it four times, runs EasyOCR with digits only, filters confidence and requires three matching sampled observations. It emits `ammo_changed`, not `shot_fired`, because a number change does not prove the cause.
>
> The minimap channel uses HSV yellow thresholding, contour area and circularity to find one waypoint. It returns coordinates normalized relative to the minimap ROI, not global world coordinates. Those observations are currently used in analysis and the demo rather than the primary event JSON.
>
> For evaluation, object detections use class-aware one-to-one matching at IoU 0.5. Overall detection F1 was about 0.976 on 167 instances. Event evaluation uses same-type matching within half a second, and the overall event F1 was only 0.5. That difference is important: tracking fragmentation and OCR smoothing affect events even when sampled-frame detection is strong. I see the project as an interpretable prototype, not a general PUBG benchmark.

---

# 4. 30-Second Project Summary

> I built a modular pipeline that converts PUBG Mobile video into structured gameplay events. YOLO11n detects characters and vehicles, semantic filtering and NMS clean the detections, an IoU tracker maintains IDs, and a temporal engine emits confirmed appearance and disappearance events. A separate EasyOCR channel recognizes ammo changes, and an HSV-contour module localizes a yellow minimap marker. Detection F1 was 0.976 on a small targeted set, while event F1 was 0.5, mainly because of tracking fragmentation and OCR recall. The system is an interpretable clip-specific prototype, not a general production model.

---

# 5. 10 Things I Absolutely Must Know

1. Exact data flow: YOLO → mapping/filtering/NMS → IoU tracking → event engine → JSON/CSV.
2. Detection, tracking and event recognition are different layers.
3. Thresholds: confidence `0.25`, semantic NMS `0.3`, tracker IoU `0.1`, evaluation IoU `0.5`.
4. Temporal settings: 5-frame confirmation, 15-frame disappearance, 30-frame retention.
5. Why `max_missing_frames >= disappearance_frames`.
6. Detection results: 163 TP, 4 FP, 4 FN, F1 `0.9760`.
7. Event results: 4 TP, 4 FP, 4 FN, F1 `0.5`.
8. OCR confirms three sampled readings and skipped intermediate ammo values.
9. Minimap positions are ROI-relative, not world coordinates, and are not serialized.
10. This is a one-clip prototype with configured ROIs/windows—not a general benchmark.

---

# 6. Key Equations

## Intersection over Union

$$IoU=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}$$

## Precision

$$P=\frac{TP}{TP+FP}$$

## Recall

$$R=\frac{TP}{TP+FN}$$

## F1

$$F1=\frac{2PR}{P+R}$$

## Circularity

$$C=\frac{4\pi A}{P^2}$$

---

# 7. Project Weaknesses / Red Flags

## 7.1 Very small, single-clip evaluation

**Why challenged:** 55 detection frames, 167 instances and only eight scored event GT records are insufficient for general claims.

**Defensible answer:**

> I treat this as targeted system validation, not a benchmark. The next step would be independent clips and held-out testing.

**Do not claim:** production accuracy or cross-game generalization.

## 7.2 Prediction-assisted annotations

**Why challenged:** Initial predictions may bias the annotator toward accepting model boxes or missing model omissions.

**Defensible answer:**

> The boxes were manually reviewed and corrected, and misses were added, but prediction assistance still creates a possible confirmation-bias risk that I disclose.

**Do not claim:** completely blind independent annotation.

## 7.3 Thresholds and ROIs are clip-specific

**Why challenged:** Strong results may depend on hand-tuning to the evaluated clip.

**Defensible answer:**

> The prototype intentionally makes these assumptions explicit and configurable. A stronger study would tune on separate validation data and report final results on held-out clips.

**Do not claim:** universal thresholds.

## 7.4 Event F1 is only 0.5

**Why challenged:** The final task is event detection, so frame-level detection scores can distract from weak system performance.

**Defensible answer:**

> The final metric honestly exposes the main weaknesses: character lifecycle fragmentation and OCR recall. I retained the result without retrospective threshold tuning.

**Do not claim:** the event system is highly accurate.

## 7.5 Initialization exclusions could look like metric manipulation

**Why challenged:** Six predictions are removed from scoring.

**Defensible answer:**

> The same type-agnostic rule is applied before matching using a window derived from confirmation frames and source FPS. All excluded items are auditable. However, it is still a protocol choice driven by montage boundaries and should be reported clearly.

**Do not claim:** exclusions are unquestionably neutral.

## 7.6 Event evaluator ignores ammo values

**Why challenged:** `ammo_changed` can match another `ammo_changed` even if the predicted values are wrong.

**Defensible answer:**

> Correct. The current evaluator checks event type and timestamp only. A stronger evaluator should also validate previous/current values or define an acceptable transition rule.

**Do not claim:** correct ammo transition semantics were evaluated.

## 7.7 IoU tracker is fragile

**Why challenged:** No Kalman filter, appearance embedding or global assignment.

**Defensible answer:**

> It is a transparent baseline that behaved better than the tested ByteTrack configuration in these short windows. I do not claim general superiority.

**Do not claim:** robust long-term identity tracking.

## 7.8 Semantic NMS can worsen localization

**Why challenged:** The failure analysis shows that a coarse high-confidence box can suppress a tight box.

**Defensible answer:**

> Semantic merging solved car/truck duplicates but introduced a localization trade-off for nested character boxes. The failure is documented rather than hidden.

**Do not claim:** NMS always selects the geometrically best box.

## 7.9 Vehicle filter caused a real FN

**Why challenged:** The heuristic directly removed a visible UAZ.

**Defensible answer:**

> It was a precision-recall choice. It removed implausible boxes but rejected one real case. I would retune it only on separate validation data.

**Do not claim:** the filter is harmless.

## 7.10 Main minimap integration discards observations

**Why challenged:** `run_pipeline` calls `analyze` but ignores the return value.

**Defensible answer:**

> Minimap localization is currently an independent analysis and visualization channel. It is not part of the event schema. Production integration needs explicit observation/event semantics.

**Do not claim:** primary outputs include minimap localization.

## 7.11 Frame-based temporal thresholds depend on FPS

**Why challenged:** Five and fifteen frames mean different durations at different FPS.

**Defensible answer:**

> The reference clip has stable 30.0023 FPS. For general use I would express thresholds in seconds and convert them using source FPS.

**Do not claim:** thresholds behave identically at every FPS.

## 7.12 `purpose` does not control processing

**Why challenged:** Segment names such as vehicle and character may imply class-specific execution, but the main loop runs the configured detector and OCR across active segments.

**Defensible answer:**

> In the current implementation, purpose documents the segment and appears in logging. It does not route models or filter classes.

**Do not claim:** segment purpose changes inference behavior.

---

# 8. Full Detailed Interview Question Bank

## Part 1 — Project Overview

## Q1. What problem does this project solve?

### Question

“What problem does your project solve?”

### What interviewer is testing

你能否把项目讲成一个明确的问题，而不是简单罗列 YOLO、OCR 等技术。

### Spoken English answer

> My project converts gameplay video into structured, time-stamped events. Instead of treating every frame independently, it detects characters and vehicles, tracks them across frames, and reasons about when an object has appeared or disappeared. It also has separate HUD and minimap channels: OCR observes magazine-ammo changes, while an HSV-based method localizes a yellow minimap marker. The final event stream is written to JSON and CSV for later analysis.

### Chinese explanation

输入是非结构化视频，输出是结构化事件。核心价值不是“画检测框”，而是把逐帧视觉观测转化为可统计、查询和评估的事件。

### Follow-up questions and answers

**Is this a general gameplay understanding system?**

> It is an engineering prototype, not a general gameplay-understanding system. The object pipeline is reusable, but the mappings, ROIs, thresholds and analysis windows are configured for this PUBG Mobile clip.

**Does it detect kills or shots?**

> It does not detect kills. It emits `ammo_changed`, not `shot_fired`, because a magazine-number change could also come from reloading, switching weapons, OCR error or an edit.

**Does it process the whole video?**

> The reference configuration processes selected continuous windows rather than claiming full-video coverage.

### Relevant implementation

- `README.md`
- `src/main.py`
- `config.example.json`

### Dangerous answers to avoid

- “It understands everything happening in PUBG.”
- “It detects shooting.”
- “It identifies enemies.”
- “It was validated across multiple games.”

## Q2. Walk me through the architecture.

### Question

“Can you walk me through the system from input video to output events?”

### What interviewer is testing

系统级表达能力，以及是否真正理解模块之间的数据流。

### Spoken English answer

> OpenCV reads the video one frame at a time and calculates the timestamp from the frame index and source FPS. For the object branch, YOLO11n detects `person`, `car` and `truck`. I map those labels to `character` and `vehicle`, apply geometry filters and semantic NMS, and then pass the detections to a class-aware IoU tracker. The tracker assigns persistent IDs. The event engine confirms a track after five consecutive observations and emits a disappearance after fifteen missing frames.
>
> In parallel, the HUD branch crops a normalized ammo ROI, enlarges it, runs EasyOCR with a digit allowlist, applies confidence filtering and requires three matching observations before confirming a value change.
>
> The minimap branch crops another ROI and uses HSV thresholding, contour area and circularity to locate a yellow marker. Lifecycle and ammo events are serialized to JSON and CSV. Minimap observations are currently analyzed and visualized separately.

### Chinese explanation

三条视觉通道：

```text
Video
├─ YOLO → mapping/filtering/NMS → tracking → temporal events → JSON/CSV
├─ HUD ROI → OCR → temporal validation → ammo_changed → JSON/CSV
└─ Minimap ROI → HSV/contours → ROI-relative marker position → demo/analysis
```

### Follow-up questions and answers

**Why separate the channels?**

> The three channels represent different visual problems. General objects suit a learned detector, fixed HUD digits suit OCR, and a distinctive fixed-color marker can be handled by a transparent color-and-shape heuristic.

**Does `purpose` select which detector classes run?**

> In the current main pipeline, the analysis segment’s `purpose` is mainly descriptive and used in logging. It does not dynamically restrict YOLO classes.

**Is minimap output serialized?**

> No. `MinimapAnalyzer.analyze` returns observations, but `run_pipeline` currently does not append them to the primary event list. The demo consumes and visualizes them separately.

### Relevant implementation

- `run_pipeline` in `src/main.py`
- `src/demo.py`
- `src/output.py`

### Dangerous answers to avoid

- “All three channels produce JSON events.”
- “Each analysis segment runs only the class named by its purpose.”
- “The minimap coordinates are world positions.”

## Q3. What is the difference between detection, tracking and event detection?

### Question

“What is the difference between object detection, tracking and event recognition?”

### What interviewer is testing

是否区分空间观测、跨帧身份和时间语义。

### Spoken English answer

> Detection answers what and where in one frame, for example a character bounding box with confidence. Tracking connects detections over time and assigns a persistent `track_id`. Event recognition reasons over the track history—for example, only emitting `character_detected` after the same track has been observed consistently, or `character_disappeared` after it has been missing long enough.

### Chinese explanation

- 检测：单帧，类别和位置。
- 跟踪：跨帧关联身份。
- 事件：根据时间状态生成具有业务意义的离散记录。

### Follow-up questions and answers

**Why not emit an event for every detection?**

> Emitting every frame would create many duplicate events and amplify one-frame false positives.

**Can tracking work without detection?**

> This tracker is tracking-by-detection, so it depends on detector boxes. It does not independently predict motion with a Kalman filter or optical flow.

### Relevant implementation

- `src/detector.py`
- `src/tracker.py`
- `src/event_engine.py`

### Dangerous answers to avoid

不要把检测框、track 和 event 当作同一层输出。

## Q4. Which parts are reusable and which are game-specific?

### Question

“What is general and what is specific to this clip or game?”

### What interviewer is testing

是否诚实理解系统的泛化边界。

### Spoken English answer

> The interfaces, typed data structures, video loop, IoU calculation, tracker, lifecycle state machine, serializers and evaluators are broadly reusable. The detector class mapping is configurable.
>
> The current semantic labels, area and aspect-ratio limits, confidence thresholds, analysis windows, ammo ROI, minimap ROI, HSV range and contour limits are game- or clip-specific. Applying it to another game would require recalibration and possibly new models.

### Chinese explanation

“算法骨架可复用”不等于“参数和视觉语义可直接泛化”。

### Follow-up questions and answers

**Would normalized ROIs solve resolution changes?**

> Normalization helps when the HUD layout scales proportionally, but it does not solve a moved HUD, a changed aspect ratio, UI scaling or a different layout.

**Would another PUBG clip work automatically?**

> I would not guarantee another clip without validation because camera content, editing, HUD layout and rendering can change.

### Relevant implementation

- `src/config.py`
- `config.example.json`

### Dangerous answers to avoid

- “It is game-agnostic.”
- “Normalized coordinates make it resolution-independent in all cases.”

## Part 2 — YOLO and Object Detection

## Q5. Why did you choose YOLO11n?

### Question

“Why YOLO11n?”

### What interviewer is testing

模型选择和工程权衡。

### Spoken English answer

> I needed a practical pretrained baseline without creating a custom training dataset. YOLO11n is the small variant, so it offers a reasonable speed and dependency footprint for an assessment prototype. Its COCO labels already include `person`, `car` and `truck`, which I could map into gameplay semantics. I am not claiming it is optimal for PUBG; domain shift is one of the project’s limitations.

### Chinese explanation

使用 nano 模型是速度、部署成本与准确率的权衡。它避免训练成本，但自然图像到游戏画面存在 domain shift。

### Follow-up questions and answers

**Why not train a custom detector?**

> A custom detector could improve domain accuracy, but I did not have a sufficiently large, independently split gameplay dataset. For this scope I prioritized a reproducible end-to-end system.

**Why not use a larger YOLO model?**

> A larger model might improve difficult detections but would cost more latency and memory. I would benchmark model sizes on held-out gameplay data before choosing one for production.

### Relevant implementation

- `UltralyticsDetector` in `src/detector.py`
- `model_name: yolo11n.pt`, `image_size: 960`

### Dangerous answers to avoid

- “YOLO11n is the most accurate model.”
- “It was chosen after a complete model benchmark.” The repository does not show that.

## Q6. How does YOLO work conceptually, and what does it output?

### Question

“Explain YOLO conceptually. What do you get from the model?”

### What interviewer is testing

目标检测基本功。

### Spoken English answer

> Conceptually, YOLO processes the image with a convolutional backbone and predicts object classes and bounding boxes in one detection pipeline. From Ultralytics I read each predicted class ID, class name, confidence and `xyxy` bounding-box coordinates. I then filter target source classes, map them to semantic classes and create a typed `Detection`.

### Chinese explanation

分类只回答“图中是什么”；检测同时回答“是什么”和“在哪里”。边界框回归预测位置，分类分支预测类别相关分数。

### Follow-up questions and answers

**What is `xyxy`?**

> `xyxy` is the top-left and bottom-right representation: `(x1, y1, x2, y2)`.

**What does confidence mean?**

> Confidence represents the model’s confidence in the detection. It is not a calibrated probability that guarantees correctness.

### Relevant implementation

`UltralyticsDetector.detect`, especially `box.cls`, `box.conf` and `box.xyxy`.

### Dangerous answers to avoid

不要声称 confidence 是严格校准后的真实概率。

## Q7. What is IoU? Give an example.

### Question

“What is intersection over union?”

### What interviewer is testing

检测与跟踪中最核心的几何指标。

### Spoken English answer

> IoU measures the overlap between two boxes. It is the intersection area divided by the union area. Zero means no overlap and one means identical boxes.
>
> For example, if two boxes each have area 100 and their intersection is 50, the union is `100 + 100 - 50 = 150`, so IoU is `50 / 150`, about `0.33`.

### Chinese explanation

$$IoU(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{I}{A_A+A_B-I}$$

### Follow-up questions and answers

**Where is IoU used here?**

> It is used by semantic NMS, tracker association and detection evaluation, but with different thresholds and meanings.

**Why can a visually related prediction fail at IoU 0.5?**

> A coarse box and a tight annotation may cover the same object but have too much extra area, reducing the overlap-to-union ratio below 0.5.

### Relevant implementation

- `calculate_iou` in `src/tracker.py`
- `_iou` in `src/detector.py`

### Dangerous answers to avoid

不要混淆 IoU 和 confidence。

## Q8. What is NMS, and what is semantic NMS here?

### Question

“Why is NMS necessary, and what do you mean by semantic NMS?”

### What interviewer is testing

重复检测抑制，以及映射顺序的理解。

### Spoken English answer

> A detector can produce several overlapping boxes for one object. NMS keeps a stronger box and suppresses sufficiently overlapping alternatives. In my semantic NMS, I first map source labels to gameplay labels. That means a `car` box and a `truck` box can both become `vehicle`, and then compete during NMS. This prevents one UAZ from creating two semantic vehicle detections and possibly two tracks.
>
> The implementation sorts by confidence and suppresses same-semantic-class boxes when IoU is greater than `0.3`.

### Chinese explanation

普通按原始类别做 NMS 可能不会让 `car` 和 `truck` 互相抑制。项目先把二者映射到 `vehicle`，再做同语义类别 NMS。

### Follow-up questions and answers

**Can NMS hurt?**

> Yes. In the failure analysis, a coarse controlled-character box sometimes had higher confidence and suppressed a tighter localization. That produced a paired FP and FN at evaluation IoU 0.5.

**Is it optimal matching?**

> No. It is greedy confidence-ordered suppression.

### Relevant implementation

`_semantic_nms` in `src/detector.py`.

### Dangerous answers to avoid

- “NMS always keeps the best-localized box.”
- “Semantic NMS improved every case.”

## Q9. Why map `person` to `character`, and `car`/`truck` to `vehicle`?

### Question

“Why did you choose these semantic mappings?”

### What interviewer is testing

标签语义是否有证据支持。

### Spoken English answer

> `person` becomes `character` because the pretrained detector only tells me that the visual pattern looks like a person. It cannot infer enemy versus teammate reliably. The scene also contains the controlled avatar and training targets.
>
> I merge `car` and `truck` into `vehicle` because YOLO alternated between those COCO labels for the same UAZ in nearby frames. Merging them gives a more stable gameplay-level category.

### Chinese explanation

原始模型类别不一定等于业务语义。映射应当只做有证据支撑的抽象，不能凭空推断阵营。

### Follow-up questions and answers

**Why exclude motorcycle and bus?**

> Inspection did not establish those classes as reliable true gameplay objects in the selected windows, so I excluded them rather than increasing false positives.

**How would you implement `enemy_detected`?**

> I would need an additional affiliation classifier or game-specific cues such as team markers, uniform appearance or HUD information. Renaming `character` to `enemy` would be semantically wrong.

### Relevant implementation

`target_classes` and `class_mapping` in `config.example.json`.

### Dangerous answers to avoid

“Every detected person is an enemy.”

## Q10. What do the geometry filters do?

### Question

“Explain your area and aspect-ratio filters.”

### What interviewer is testing

后处理逻辑和精确率/召回率权衡。

### Spoken English answer

> After semantic mapping, I compute bounding-box area as a fraction of the full frame and calculate width divided by height. I reject boxes outside the configured range. Character area is `0.003–0.35` with aspect ratio `0.2–1.5`. Vehicle area is `0.002–0.2` with aspect ratio `0.7–3.5`. Normalized area is more portable than hard-coded pixel dimensions.
>
> The filters increase precision but can reduce recall. The documented vehicle false negative happened because the only truck prediction was oversized and was rejected.

### Chinese explanation

几何过滤利用先验排除不合理框，但这属于启发式方法。阈值严格时 FP 减少、FN 可能增加。

### Follow-up questions and answers

**Are they learned?**

> No, they are explicit configured heuristics.

**Were they validated independently?**

> They were validated on a limited targeted set, not on a broad held-out benchmark. That is a limitation.

### Relevant implementation

`_passes_filter` in `src/detector.py`.

### Dangerous answers to avoid

不要把手工阈值说成训练获得的通用参数。

## Q11. Explain the three easily confused thresholds.

### Question

“What is the difference between confidence threshold, NMS IoU threshold and tracker IoU threshold?”

### What interviewer is testing

是否真正理解流水线。

### Spoken English answer

> The confidence threshold, `0.25`, decides whether a raw YOLO prediction is accepted at all. The semantic NMS IoU threshold, `0.3`, decides whether two overlapping same-semantic-class detections in one frame are likely duplicates. The tracker IoU threshold, `0.1`, decides whether a current detection can be associated with a previous track across frames.
>
> They operate at different stages and should not be treated as one interchangeable parameter.

### Chinese explanation

- confidence：模型分数门槛。
- NMS IoU：同一帧去重门槛。
- tracker IoU：跨帧关联门槛。

### Follow-up questions and answers

**What if confidence increases?**

> Higher confidence usually reduces FP but increases FN.

**What if NMS IoU increases?**

> Higher NMS IoU suppresses fewer overlapping boxes, so duplicates may survive.

**What if tracker IoU increases?**

> Higher tracker IoU requires more stable overlap, increasing new-track creation and possible ID fragmentation.

### Relevant implementation

`config.example.json`.

### Dangerous answers to avoid

不要说“更高 IoU threshold 总是更严格”而不说明是哪一个 IoU 阈值；对 NMS 而言，阈值升高实际上会减少抑制。

## Part 3 — Tracking

## Q12. How does your IoU tracker work?

### Question

“Explain your tracker step by step.”

### What interviewer is testing

数据关联的实际实现。

### Spoken English answer

> At each frame, the tracker first expires tracks that have been missing for more than thirty frames. It calculates IoU between every current detection and existing track of the same semantic class. Candidate pairs with IoU at least `0.1` are sorted from highest overlap to lowest. It greedily accepts a pair only if neither the detection nor the track has already been matched. Unmatched detections create new IDs. Matched tracks update their box and last-seen frame.

### Chinese explanation

这是 greedy bipartite association，不是匈牙利算法，也没有运动预测或外观 embedding。

### Follow-up questions and answers

**Why class-aware?**

> Class awareness prevents a vehicle detection from inheriting a character identity.

**What is `track_id`?**

> `track_id` is an internal persistent identity assigned by the tracker; it is not a real game entity ID.

**Can one detection match two tracks?**

> No. The greedy matched sets enforce one-to-one assignment within a frame.

### Relevant implementation

`IoUTracker.update` in `src/tracker.py`.

### Dangerous answers to avoid

- “It uses motion prediction.”
- “It finds the globally optimal association.”
- “A track ID identifies the actual player account.”

## Q13. Why use tracker IoU 0.1?

### Question

“Why is the tracking IoU threshold so low?”

### What interviewer is testing

是否能解释非标准或敏感阈值。

### Spoken English answer

> The game camera and bounding boxes can move significantly between frames, and the detector’s box shape can fluctuate. A low threshold allows the same object to remain associated despite that variation. The trade-off is a greater risk of incorrect association when nearby same-class objects overlap. It was selected for stability in these validation windows, not claimed as a universal setting.

### Chinese explanation

低门槛提高关联召回，但可能错误合并；高门槛减少错误合并，却会导致轨迹碎片化。

### Follow-up questions and answers

**How would you tune it properly?**

> I would tune it on separate validation data using ID switches, track fragmentation and downstream event quality, not only detection F1.

**Could zero be used?**

> Zero would allow non-overlapping same-class detections to become candidates, so greedy matching could associate unrelated objects. That would be dangerous.

### Relevant implementation

`iou_matching_threshold: 0.1` in `config.example.json`.

### Dangerous answers to avoid

“0.1 is the standard tracker threshold.”

## Q14. What happens during temporary disappearance?

### Question

“Suppose YOLO misses an object for ten frames. What happens?”

### What interviewer is testing

tracker retention 和 event disappearance 的区别。

### Spoken English answer

> With the current settings, the track is retained because `max_missing_frames` is thirty. The event engine also does not emit disappearance because ten is below the fifteen-frame disappearance threshold. If the object returns with enough IoU and the same class, it can recover the same ID.
>
> If the track was already confirmed, it remains confirmed and does not emit a second detected event. If it was not yet confirmed, the consecutive-observation count resets when it returns because the observations were not consecutive.

### Chinese explanation

tracker 保存身份；event engine 保存事件生命周期。短暂漏检不会立即产生消失事件。

### Follow-up questions and answers

**What happens at fifteen missing frames?**

> At fifteen missing frames, a confirmed state emits `<class>_disappeared`. `run_pipeline` then retires that ID from the tracker.

**What happens after thirty?**

> The tracker’s own expiration rule is strictly greater than thirty missing frames, but normal pipeline retirement happens earlier after the disappearance event.

**Why require `max_missing_frames >= disappearance_frames`?**

> Otherwise the tracker could delete the identity before the event engine had the chance to emit its disappearance event.

### Relevant implementation

- `IoUTracker._expire_tracks`
- `EventEngine.update`
- validation in `PipelineConfig.validate`

### Dangerous answers to avoid

不要混淆 `disappearance_frames = 15`（何时宣布事件）和 `max_missing_frames = 30`（tracker 最长保存多久）。

## Q15. What causes ID switches, and why not ByteTrack or DeepSORT?

### Question

“What are the weaknesses of IoU-only tracking?”

### What interviewer is testing

对 baseline 局限和替代方案的认识。

### Spoken English answer

> IoU-only tracking is sensitive to abrupt camera motion, detector box jitter, long occlusion and overlapping objects. Two nearby characters may exchange identities, or one object may receive a new ID after its box moves too far.
>
> I compared it with Ultralytics ByteTrack on the selected windows. My IoU tracker produced one vehicle ID and eight character IDs, while the ByteTrack run produced two vehicle IDs and fifteen character IDs and introduced an additional `lap` dependency. I retained the simpler tracker because it was more stable for this specific evaluation and easier to inspect. I do not claim it is generally better than ByteTrack.
>
> DeepSORT or an appearance-aware tracker could be stronger under long occlusion, but would add an embedding model and more complexity.

### Chinese explanation

IoU 只有空间重叠证据。ByteTrack 能利用高低分检测和运动状态；DeepSORT 还使用外观特征，因此遮挡恢复潜力更强。

### Follow-up questions and answers

**When would ByteTrack be better?**

> ByteTrack can be better when detections temporarily become low confidence but motion remains consistent, especially in longer continuous scenes.

**What happens if two characters overlap?**

> With overlap, greedy IoU may associate each new box with the wrong previous box, producing ID switches or fragmentation. There is no appearance cue to resolve ambiguity.

### Relevant implementation

`src/tracker.py`; comparison documented in `README.md`.

### Dangerous answers to avoid

- “My tracker outperforms ByteTrack in general.”
- “It handles occlusion robustly.”
- “It uses appearance.”

## Part 4 — Temporal Event Engine

## Q16. How exactly does a detection become `character_detected`?

### Question

“What exactly triggers a detected event?”

### What interviewer is testing

项目最关键的状态机逻辑。

### Spoken English answer

> A raw character detection does not immediately become an event. It first receives a track ID. The event engine stores state for that ID and counts consecutive observations. With the reference configuration, the fifth consecutive frame produces `character_detected`. The state then becomes confirmed, so later frames do not emit duplicates.

### Chinese explanation

`min_confirmation_frames = 5` 是去抖机制：只把持续观测视为真实出现。

### Follow-up questions and answers

**Does the event timestamp indicate the first frame?**

> The event timestamp is the confirmation frame, not the first observation frame.

**What if frame 3 is missed?**

> For an unconfirmed track, a gap resets `consecutive_seen` to one when it returns.

**How is deduplication achieved?**

> Each state has a `confirmed` flag, so the engine emits detected only once for that lifecycle.

### Relevant implementation

`_ObjectState` and `EventEngine.update` in `src/event_engine.py`.

### Dangerous answers to avoid

“The first YOLO box immediately creates a detected event.”

### Frame-by-frame example

| Frame | Tracker output | Event-engine state | Event |
|---:|---|---|---|
| 100 | Character ID 7 | consecutive = 1 | None |
| 101 | Character ID 7 | consecutive = 2 | None |
| 102 | Character ID 7 | consecutive = 3 | None |
| 103 | Character ID 7 | consecutive = 4 | None |
| 104 | Character ID 7 | consecutive = 5, confirmed | `character_detected` |
| 105 | Character ID 7 | already confirmed | None |
| 106–119 | ID 7 absent | missing reaches 15 at 119 | `character_disappeared` at 119 |

## Q17. What exactly triggers disappearance?

### Question

“When is `character_disappeared` emitted?”

### What interviewer is testing

off-by-one、生命周期状态和误报控制。

### Spoken English answer

> For each known event-engine state that is not present in the current tracked objects, I calculate `current_frame - last_seen_frame`. When that reaches fifteen, a disappearance event is emitted only if the track had previously been confirmed. The state is then deleted, and the main loop asks the tracker to retire that ID.

### Chinese explanation

未确认的短噪声目标不会产生 disappearance。已确认目标连续缺失达到阈值才发事件，而且只发一次。

### Follow-up questions and answers

**Why not disappear on the first miss?**

> One-frame misses are common due to blur, confidence fluctuation and occlusion. Waiting reduces false disappearance events.

**Is the event time the actual physical disappearance time?**

> It is the confirmation time after the missing window, not necessarily the exact physical instant the object left.

### Relevant implementation

`EventEngine.update` and `run_pipeline` retirement logic.

### Dangerous answers to avoid

不要称其为精确的离场时间；它带有约 15 帧确认延迟。

## Q18. Why reset state at segment boundaries?

### Question

“Why do you reset tracker, event engine and OCR at segment boundaries?”

### What interviewer is testing

视频剪辑与时间连续性的理解。

### Spoken English answer

> The source is a montage rather than one continuous match. A hard edit can make every visible object disappear instantly and unrelated objects appear. Carrying state across the edit would manufacture lifecycle events. Therefore tracker and event-engine state are discarded at segment boundaries, and OCR is reset when a new segment becomes active. A boundary reset itself does not emit disappearance because there is no evidence that the object disappeared in continuous gameplay.

### Chinese explanation

状态机默认相邻帧在同一时空连续场景。剪辑点破坏此前提，因此必须截断状态。

### Follow-up questions and answers

**Why keep initialization events?**

> Raw initialization events accurately describe pipeline startup state, so they remain in output. The evaluation excludes them using one deterministic boundary rule.

**Does reset hurt continuity?**

> Yes. Segment gating sacrifices full-video lifecycle continuity, but that is more defensible than treating montage edits as gameplay events.

### Relevant implementation

Segment handling in `src/main.py`.

### Dangerous answers to avoid

- “A cut means all objects disappeared.”
- “The system automatically detects cuts.” It uses configured windows, not automatic shot-boundary detection.

## Part 5 — OCR

## Q19. Explain the full OCR pipeline.

### Question

“How do you recognize ammo changes?”

### What interviewer is testing

OCR preprocessing、置信度和时序验证。

### Spoken English answer

> I crop a normalized fixed HUD ROI for the magazine counter. At 1920 by 1080 it corresponds to approximately pixels `688:715` horizontally and `1025:1068` vertically. I sample every five frames and upscale the crop four times using bicubic interpolation. EasyOCR runs with an English reader, greedy decoding and a digit-only allowlist. I concatenate recognized text, parse the numeric characters and reject EasyOCR observations below `0.25` confidence. A value must appear in three matching sampled observations before it is confirmed. The initial confirmed value creates no event; later confirmed changes create `ammo_changed`.

### Chinese explanation

```text
normalized ROI → pixel crop → 4× bicubic resize → EasyOCR digits only
→ numeric parsing → confidence ≥ 0.25 → 3 consistent sampled observations
→ ammo_changed
```

### Follow-up questions and answers

**Why fixed ROI?**

> The counter is in a fixed HUD location in the selected clip, so ROI cropping removes irrelevant pixels.

**Why upscale?**

> The original crop is small; upscaling gives the OCR model a larger representation, although it cannot recreate missing detail.

**Why digit allowlist?**

> The expected output is numeric, so the allowlist reduces character confusion.

**Why EasyOCR?**

> EasyOCR was the reference backend and provides confidence values. Tesseract support exists, but the reported results use EasyOCR.

### Relevant implementation

`OCRProcessor.process` and `_update_value` in `src/ocr.py`.

### Dangerous answers to avoid

- “Upscaling adds real information.”
- “OCR runs on every frame.” It samples every five frames.
- “Three consecutive video frames.” They are three sampled OCR observations.

## Q20. How does OCR temporal smoothing behave?

### Question

“What happens if OCR alternates between 10 and 18?”

### What interviewer is testing

候选状态机是否理解准确。

### Spoken English answer

> The candidate changes whenever the parsed value changes. Its count goes back to one, so alternating `10, 18, 10, 18` never reaches three matching observations and generates no change event. An invalid or low-confidence observation clears the candidate and count. This reduces false events but can miss short-lived real values.

### Chinese explanation

当前实现要求连续三次有效采样值完全一致，不是多数投票，也不是滑动窗口投票。

### Follow-up questions and answers

**How is event confidence calculated?**

> For EasyOCR, event confidence is the minimum confidence among the last three confirming observations.

**Does the first stable reading emit an event?**

> No. The first stable value initializes state; only later changes emit events.

**What happened in the reference run?**

> There were 43 sampled observations, 36 parseable and 34 above the threshold. The pipeline emitted three stable changes: `14→12`, `12→10` and `10→8`, skipping intermediate values.

### Relevant implementation

`_ValueState` and `_update_value` in `src/ocr.py`.

### Dangerous answers to avoid

“Temporal smoothing improves both precision and recall.” Here it improves stability at a clear recall/latency cost.

## Q21. Why `ammo_changed`, not `shot_fired`?

### Question

“Can you infer shooting from an ammo decrease?”

### What interviewer is testing

是否过度解释观测。

### Spoken English answer

> Not reliably from this implementation. The direct observation is that the displayed magazine value changed. Reloading, switching weapon, montage editing or OCR errors can also change that value. So I deliberately named the event `ammo_changed`. A reliable `shot_fired` event would need extra evidence such as muzzle flash, audio, recoil or game telemetry.

### Chinese explanation

视觉观测与因果事件必须分开。数字变化不唯一对应开枪。

### Follow-up questions and answers

**What causes OCR errors?**

> Small text, compression, motion, low contrast and Spanish subtitles overlapping the lower HUD are documented causes.

**How would you improve robustness?**

> For multiple games I would first locate the HUD dynamically, normalize layout and resolution, and consider a HUD-specific digit classifier trained on representative crops. I would validate on held-out clips.

### Relevant implementation

Event construction in `OCRProcessor._update_value`.

### Dangerous answers to avoid

“Every decrement is one shot.”

## Part 6 — Minimap

## Q22. How does minimap localization work?

### Question

“Explain your minimap marker detector.”

### What interviewer is testing

HSV、轮廓、形状特征和坐标系。

### Spoken English answer

> The analyzer crops a configured minimap ROI. The marker detector converts the crop from BGR to HSV and thresholds yellow pixels using lower `(22,120,120)` and upper `(38,255,255)`. It extracts external contours, filters them by area `900–2200` and circularity at least `0.45`, and chooses the largest surviving contour. The centroid comes from image moments. Finally, the center is divided by crop width and height to produce normalized ROI-relative coordinates between zero and one.

### Chinese explanation

HSV 将色相与亮度部分分离，通常比 RGB 更容易表达“黄色范围”。圆度公式是 $C=4\pi A/P^2$，理想圆接近 1，不规则形状较低。质心为 $c_x=m_{10}/m_{00}$、$c_y=m_{01}/m_{00}$。

### Follow-up questions and answers

**What is a contour?**

> A contour is a boundary representation of a connected shape in the threshold mask.

**What is returned when there is no marker?**

> It returns an empty observation list.

**Are these world coordinates?**

> No. They are positions relative to the on-screen minimap crop, not calibrated game-world coordinates.

### Relevant implementation

`HSVMarkerDetector.detect` and `MinimapAnalyzer.analyze` in `src/minimap.py`.

### Dangerous answers to avoid

- “It understands the map.”
- “It returns global location.”
- “The confidence is learned.” It is `min(1, circularity)`.

## Q23. Why is minimap output not in primary JSON/CSV?

### Question

“Why doesn’t minimap localization appear in the event output?”

### What interviewer is testing

能否准确区分当前实现和未来方向。

### Spoken English answer

> The current minimap result is a frame-level observation rather than a lifecycle event with defined semantics. I evaluated and visualized it separately. In `run_pipeline`, `analyze` is called when enabled, but its return value is not appended to the event list. The demo directly consumes the observations. A production version should define a schema and temporal semantics before serializing them.

### Chinese explanation

连续位置 observation 和离散 event 是不同数据类型。当前项目没有定义 marker appeared/moved/disappeared 等状态逻辑。

### Follow-up questions and answers

**Is minimap enabled in the reference config?**

> It is disabled in the main reference configuration, although its ROI and thresholds are present. The dedicated demo/analysis uses it.

**What were its results?**

> In the selected validation samples it detected the marker in 76 out of 76 positive frames and produced no false positives in 178 sampled negative frames. That is only clip-specific validation.

### Relevant implementation

- `minimap.enabled: false`
- `run_pipeline` in `src/main.py`
- minimap section in `src/demo.py`

### Dangerous answers to avoid

“Minimap events are exported alongside ammo and object events.”

## Part 7 — Evaluation

## Q24. How did you evaluate object detection?

### Question

“Describe your detection evaluation protocol and results.”

### What interviewer is testing

指标是否可信、匹配是否正确。

### Spoken English answer

> I used 55 unique annotated frames containing 167 ground-truth instances: 142 characters and 25 vehicles. Predictions and annotations must have the same frame and semantic class and IoU at least `0.5`. I build all eligible pairs, sort by descending IoU and greedily match one prediction to at most one ground-truth box. Unmatched predictions are false positives and unmatched annotations are false negatives.
>
> Character results were 139 TP, 4 FP and 3 FN, giving precision `0.9720`, recall `0.9789` and F1 `0.9754`. Vehicle results were 24 TP, 0 FP and 1 FN, giving precision `1.0`, recall `0.96` and F1 `0.9796`. Overall there were 163 TP, 4 FP and 4 FN, with precision, recall and F1 all about `0.9760`.

### Chinese explanation

一对一匹配很重要。否则多个预测框可能重复“命中”一个 GT，虚增 TP。

### Follow-up questions and answers

**Why IoU 0.5?**

> `0.5` is a conventional, interpretable threshold for a basic targeted evaluation, but reporting multiple IoU thresholds would be more complete.

**Is greedy matching always optimal?**

> Greedy matching is deterministic and adequate here, but it is not guaranteed to be a globally optimal assignment in every arrangement.

**Why do class frame counts add to 57, not 55?**

> Two frames contain both classes, so the per-class frame counts overlap.

### Relevant implementation

- `evaluate_detections` in `src/evaluator.py`
- `outputs/evaluation/detection_metrics.json`
- `annotations.json`

### Dangerous answers to avoid

- “This is benchmark-level accuracy.”
- “The dataset is independent and large.”
- “Greedy matching is globally optimal.”

## Q25. Define precision, recall and F1.

### Question

“What do precision, recall and F1 mean?”

### What interviewer is testing

评价指标基础。

### Spoken English answer

> Precision asks: of everything I predicted, how much was correct? Recall asks: of everything that existed in ground truth, how much did I find? F1 is the harmonic mean that balances the two.
>
> For example, with 8 true positives, 2 false positives and 4 false negatives, precision is `8/10 = 0.8`, recall is `8/12 ≈ 0.667`, and F1 is about `0.727`.

### Chinese explanation

$$Precision=\frac{TP}{TP+FP},\quad Recall=\frac{TP}{TP+FN},\quad F1=\frac{2PR}{P+R}$$

### Follow-up questions and answers

**When would you prioritize precision?**

> Precision matters when false alarms are costly.

**When would you prioritize recall?**

> Recall matters when missing an important event is costly. The product requirement should determine the operating point.

### Relevant implementation

`Metrics.from_counts` in `src/evaluator.py`.

### Dangerous answers to avoid

不要说 accuracy 和 precision 是同一个指标。

## Q26. How did you evaluate events?

### Question

“How does event evaluation work, and what were the results?”

### What interviewer is testing

时间匹配、边界处理、事件效果与检测效果的差异。

### Spoken English answer

> Event predictions match ground truth one-to-one if the event type is the same and their timestamps differ by at most `0.5` seconds. Candidate pairs are sorted by the smallest time difference.
>
> After boundary exclusion, there were eight scored predictions and eight ground-truth events: 4 TP, 4 FP and 4 FN, so precision, recall and F1 were all `0.5`.
>
> `character_detected` had F1 `0.5`, `character_disappeared` `0.6667`, and `ammo_changed` `0.4444`. `vehicle_detected` had zero scored predictions and zero scored GT, so its metrics are not meaningful even though the JSON stores zeros.

### Chinese explanation

检测评估看单帧框；事件评估看时间状态机最终是否在正确时间产生正确事件。后者会累积 detection、tracking、OCR 和 temporal logic 的误差。

### Follow-up questions and answers

**Does ammo matching check previous/current values?**

> No. The evaluator checks only event type and timestamp. It does not validate `previous_value` and `current_value`; that is a limitation.

**Why is event F1 much lower?**

> Character track fragmentation created extra lifecycle events, and OCR smoothing skipped intermediate ammo values. High sampled-frame detection F1 therefore did not translate into high event F1.

**Does object ID have to match?**

> No. Event matching does not use object ID.

### Relevant implementation

- `evaluate_events` in `src/evaluator.py`
- `outputs/evaluation/event_metrics.json`
- `events_ground_truth.json`

### Dangerous answers to avoid

- “The event evaluator validates ammo values.”
- “Event F1 is high.”
- “Detection F1 proves the complete system is accurate.”

## Q27. Explain the initialization exclusion window. Is it metric manipulation?

### Question

“Why did you exclude initialization events, and could that be considered cherry-picking?”

### What interviewer is testing

评价协议是否严谨、是否诚实。

### Spoken English answer

> At the start of a configured segment, objects can already be visible. After five confirming frames, the state machine emits detected events, but those are pipeline initialization rather than objects entering during continuous gameplay. I exclude every prediction of every event type in the interval from segment start through `min_confirmation_frames / FPS`. Here that is `5 / 30.0023 = 0.166654` seconds.
>
> The rule is applied before matching, is independent of whether an event would help or hurt the metric, and all six excluded predictions are retained in an audit file. One GT vehicle event at the montage boundary is also recorded separately.
>
> It could still be challenged because the protocol was designed for a montage and the dataset is small. The defensible point is transparency and deterministic application—not that exclusions are inherently beyond criticism.

### Chinese explanation

初始化事件与“真实进入场景”语义不同。关键是规则预先定义、全类型统一、匹配前执行、保留审计。

### Follow-up questions and answers

**Why exactly five frames?**

> Five comes from the event engine’s configured confirmation threshold, so the window corresponds to expected initialization latency.

**Could you avoid exclusion entirely?**

> A cleaner production protocol could label segment-start occupancy separately from true entry events, or evaluate only naturally continuous streams without montage boundaries.

### Relevant implementation

- `partition_segment_initialization_events`
- `outputs/evaluation/excluded_boundary_events.json`

### Dangerous answers to avoid

- “Exclusion has no effect on the metric.”
- “It proves there is no bias.”
- Hiding that six predictions were excluded.

## Q28. What are the evaluation dataset limitations?

### Question

“How much confidence should we have in these metrics?”

### What interviewer is testing

科研诚信和统计意识。

### Spoken English answer

> They demonstrate targeted functionality on one documented clip, not generalization. The annotation set has only 55 unique frames and 167 instances, and the event ground truth has eight scored events. Initial boxes were prediction-assisted and then manually reviewed and corrected, so this is not a fully independent blind test set. The ROIs and thresholds are also specific to selected windows. I would need separate train, tuning and held-out test clips across maps, scenes and UI conditions before making broader claims.

### Chinese explanation

样本小、单视频、调参与评价可能共享数据、稀有事件少，都会让指标方差大且容易乐观。

### Follow-up questions and answers

**Is prediction-assisted annotation invalid?**

> It is not automatically invalid if every box is manually reviewed and misses are added, but it risks confirmation bias and should be disclosed.

**What would a stronger protocol look like?**

> I would use independent clips, separate threshold tuning from final testing, include multiple gameplay conditions and report confidence intervals and per-scenario metrics.

### Relevant implementation

README evaluation protocol and annotation files.

### Dangerous answers to avoid

“The results prove production readiness.”

## Part 8 — Failure Analysis

## Q29. What were the main detection failures?

### Question

“Explain your representative FP and FN cases.”

### What interviewer is testing

是否进行了真正的误差分析。

### Spoken English answer

> The character errors mainly involved the controlled avatar. YOLO produced nested or coarse boxes. Because semantic NMS is confidence-ordered, it could retain the coarse box and suppress a tighter box. At IoU 0.5 the retained coarse box became an FP while the tighter ground-truth object became an FN. This happened in frames such as 2370 and 2384.
>
> The vehicle FN was frame 1123. A UAZ was visible, but YOLO only produced an oversized `truck` prediction overlapping the scope. The vehicle area filter rejected it, so the final result contained no matching vehicle prediction.

### Chinese explanation

- Character case：模型候选 + NMS 选择共同导致。
- Vehicle case：模型没有产生合适框，后处理又拒绝唯一粗框。
- 同一物体可能同时造成一个 FP 和一个 FN，因为框定位不满足 IoU。

### Follow-up questions and answers

**Which failure is model versus post-processing?**

> The coarse or oversized candidates originate from the pretrained model under domain shift. Selecting or rejecting those candidates is a post-processing effect.

**How would you fix them?**

> I would first collect more representative annotations, inspect threshold sensitivity on held-out data and fine-tune a gameplay detector. For nested character boxes, soft-NMS or task-aware selection could be tested. For the vehicle filter, I would avoid simply loosening it on the same sample without checking the false-positive cost.

### Relevant implementation

`outputs/analysis/failures/failure_cases.json`.

### Dangerous answers to avoid

- Blaming every error entirely on YOLO.
- Claiming that removing the filter is definitely the solution.
- Tuning directly against the test failure and reporting the same test score.

## Part 9 — Engineering and Software Design

## Q30. Why modularize the project?

### Question

“Why did you split the code into these modules?”

### What interviewer is testing

软件设计、可测性、关注点分离。

### Spoken English answer

> Detection, tracking, temporal reasoning, OCR and minimap analysis have different responsibilities and failure modes. Keeping them separate lets me test them using synthetic detections or observations without loading YOLO or processing an entire video. Shared typed dataclasses define the boundaries, and configuration keeps clip-specific parameters out of the core logic.

### Chinese explanation

模块化使单元测试、替换组件、定位错误更容易，也避免一个巨型脚本把模型推理和业务状态混在一起。

### Follow-up questions and answers

**What does each module do?**

> `detector` adapts YOLO; `tracker` assigns identities; `event_engine` creates lifecycle events; `ocr` validates HUD values; `minimap` localizes a marker; `evaluator` scores outputs; `output` serializes events; `visualizer` and `demo` render results; `config` loads and validates typed settings.

**What do tests cover?**

> The 31 tests cover validation, semantic mapping and NMS, IoU tracking, lifecycle confirmation and disappearance, one-to-one evaluation, boundary exclusion, OCR smoothing, minimap normalization, serialization and invalid video metadata.

### Relevant implementation

`src/` and `tests/`.

### Dangerous answers to avoid

“The tests validate real YOLO accuracy.” Most are unit-level behavioral tests.

## Q31. How is reproducibility handled?

### Question

“How can another engineer reproduce your result?”

### What interviewer is testing

工程完整性。

### Spoken English answer

> The README documents the exact public video, attribution, local filename, SHA-1, byte size, dimensions, FPS and analysis windows. Dependencies have bounded major versions, thresholds and ROIs are in JSON, output artifacts and annotations are included, and the evaluation protocol is documented. The main limitation is that first-time YOLO and EasyOCR model downloads require network access unless weights are already cached.

### Chinese explanation

可复现不仅是代码，还包括输入版本、依赖、配置、评价规则和结果产物。

### Follow-up questions and answers

**Are dependencies fully pinned?**

> Dependencies are range-bounded, not locked to exact builds, so this is useful but not perfect reproducibility.

**Is execution deterministic?**

> I would not promise bit-for-bit determinism across library versions and hardware. A production setup should pin exact artifacts and record model hashes.

### Relevant implementation

`requirements.txt` and `README.md`.

### Dangerous answers to avoid

“The environment is perfectly reproducible.”

## Q32. What are the performance bottlenecks?

### Question

“What was runtime, and how would you improve throughput?”

### What interviewer is testing

性能分析和实时系统思维。

### Spoken English answer

> On an AMD Ryzen 9 7945HX under WSL2 using CPU only, the evaluated configuration processed 377 detector-analyzed frames in 38.52 seconds, about 9.79 analyzed frames per second. Peak reported memory was about 2.24 GB. YOLO inference is likely the main cost, with video decoding and OCR also contributing.
>
> For higher throughput I would benchmark GPU inference, use mixed precision where supported, batch frames for offline processing, reduce image size or use frame skipping when acceptable, and profile before optimizing. For live streaming I would use bounded queues and control latency rather than allowing backlog to grow.

### Chinese explanation

离线吞吐率与实时延迟不同。批处理能提高吞吐，但可能增加实时延迟。

### Follow-up questions and answers

**Was the system real-time?**

> Not at the original 30 FPS on CPU.

**Why is 9.79 not source FPS?**

> The reported rate is based on frames where detector analysis ran in the configured windows, not all decoded frames.

**What does “analyzed frame” mean?**

> The run decoded 2,581 frames through the last selected window, but YOLO analyzed 377 frames inside the two windows.

### Relevant implementation

README Performance section.

### Dangerous answers to avoid

- “It runs at real-time 30 FPS.”
- Comparing 9.79 analyzed FPS directly with full-video decoding without qualification.

## Q33. What would you change for production?

### Question

“How would you productionize this?”

### What interviewer is testing

是否能从 prototype 过渡到产品架构。

### Spoken English answer

> I would first define the event semantics and product error costs. Then I would collect multi-game, multi-device data with independent train, validation and test splits. I would fine-tune or replace the detector, use a stronger tracker with motion and possibly appearance, and replace fixed ROIs with UI detection or per-game calibration. Temporal thresholds should be expressed in time rather than raw frame counts. I would add event-value-aware evaluation, monitoring, model/version metadata, latency budgets and graceful degradation.
>
> That is future work. The current repository is a transparent offline prototype with configured segments and heuristics.

### Chinese explanation

生产化不是单纯换更大模型，还需要数据闭环、服务可靠性、指标定义、漂移监控和版本管理。

### Follow-up questions and answers

**How would live processing differ?**

> Live processing needs asynchronous capture, bounded buffers, latency monitoring and state recovery.

**How would you support five games?**

> I would keep shared interfaces but create per-game adapters for semantic mapping, HUD localization, OCR and minimap interpretation, selected through a game profile.

### Relevant implementation

This is a proposed production design, not existing functionality.

### Dangerous answers to avoid

不要用将来方案冒充当前实现。

## Part 10 — Design Trade-offs

## Q34. Why not segmentation, optical flow or a vision-language model?

### Question

“Why did you choose this pipeline over more sophisticated approaches?”

### What interviewer is testing

是否根据问题选择工具，而非堆砌模型。

### Spoken English answer

> Bounding boxes were sufficient for the lifecycle events I wanted, so segmentation would add annotation and compute cost without being necessary for the current output. Optical flow could help motion continuity but does not by itself provide semantic classes and can struggle with camera motion and cuts. A vision-language model might provide richer semantics, but it would be slower, harder to evaluate deterministically and unnecessary for fixed categories and HUD digits. For this assessment I preferred inspectable components with explicit failure modes.

### Chinese explanation

技术选择应由输出需求驱动。更复杂不等于更合适。

### Follow-up questions and answers

**When would segmentation help?**

> Segmentation would help if precise silhouettes, occlusion reasoning or screen-area coverage were required.

**When would a VLM help?**

> A VLM could help with open-ended scene descriptions or previously unseen UI semantics, but would need careful latency and reliability evaluation.

### Relevant implementation

Current repository implements detection boxes, not segmentation, flow or VLM.

### Dangerous answers to avoid

- “Those methods are worse.”
- “YOLO solves all semantic understanding.”

## Q35. Why heuristics for minimap and OCR rather than custom learned models?

### Question

“Why use HSV and EasyOCR instead of training dedicated models?”

### What interviewer is testing

需求规模、数据成本和可解释性权衡。

### Spoken English answer

> The yellow waypoint has a distinctive color and approximate shape in a fixed ROI, so HSV and contour constraints provide a small, interpretable baseline without a marker dataset. The ammo counter is also a narrow OCR problem, and EasyOCR allowed me to test feasibility without training labels.
>
> These choices are appropriate for this prototype but fragile across UI themes and games. With enough production data, a marker detector and HUD-specific digit classifier could generalize better.

### Chinese explanation

启发式方法在约束明确、数据不足时成本低，但分布变化时通常不稳健。

### Follow-up questions and answers

**What breaks in another game?**

> Color, marker shape, HUD position, font, resolution, semantic labels and thresholds may all change.

**What remains reusable?**

> The interfaces, normalized crop utility, temporal state logic, tracker, event schema and evaluator structure remain reusable.

### Relevant implementation

`src/minimap.py` and `src/ocr.py`.

### Dangerous answers to avoid

“Heuristics generalize better because they are simple.”

---

# 9. Live Modification Scenarios

Use this spoken structure: **current behavior → failure/risk → proposed change → validation**.

| Scenario | Strong structured answer |
|---|---|
| Want `enemy_detected` | Current model only outputs `person`, mapped to `character`. I would not rename it. I would add a separately evaluated affiliation stage using team-color/UI or trained appearance cues, then emit enemy only when that stage is confident. |
| Two characters overlap | IoU-only greedy association may swap IDs or fragment tracks. I would measure ID switches and test motion prediction plus appearance embeddings, while preserving class-aware one-to-one matching. |
| FPS changes 30→60 | Frame-count thresholds halve their duration: 5 frames changes from about 167 ms to 83 ms; 15 changes from 500 ms to 250 ms. I would define thresholds in seconds and convert using measured FPS. |
| Resolution changes | Normalized ROIs scale, but font size, aspect ratio and UI scaling can still differ. I would validate ROI placement and potentially detect HUD anchors dynamically. Geometry filters already use normalized area. |
| HUD moves | Fixed ROI reads the wrong pixels. I would locate the HUD first using a template, detector or calibration profile, then crop relative to the detected anchor. |
| Minimap marker changes color | The configured HSV range fails. I would use a per-game color profile or train a marker detector if appearance varies significantly. |
| YOLO misses object for 10 frames | Existing confirmed ID can survive: 10 is below disappearance 15 and retention 30. If it returns with enough IoU, no duplicate detected event is emitted. |
| Need real-time inference | Profile first, use GPU/mixed precision, consider lower resolution/frame sampling, decouple capture and inference with bounded queues, and monitor end-to-end latency. |
| Support five games | Introduce game profiles/adapters for classes, ROIs, HUD parsers and marker semantics, while sharing tracking, event schema, serialization and evaluation infrastructure. |
| Detection confidence unstable | Calibrate threshold on held-out data, use temporal evidence, potentially retain lower-confidence boxes in a stronger tracker, and monitor precision/recall rather than reacting frame by frame. |
| OCR alternates 10/18 | Current logic never confirms either because each change resets the count. For production I might use a time-window voting or probabilistic sequence model, but I would test latency and skipped-transition behavior. |
| Production redesign | Add diverse data, held-out evaluation, stronger tracker, per-game calibration, event-value-aware scoring, time-based thresholds, monitoring, model versioning and live-stream backpressure. |

## Dangerous answers for live-change questions

- “I would just lower the threshold.”
- “I would use a bigger model” without defining data, metric and latency effects.
- “Normalized coordinates solve every resolution issue.”
- “A new label name makes characters into enemies.”
- “At 60 FPS everything behaves the same.”

---

# 10. CV Fundamentals Derived from This Project

| Concept | Project-specific explanation |
|---|---|
| CNN | A neural network that learns spatial visual features. YOLO uses convolutional feature extraction to detect people and vehicles. |
| Convolution | A learned kernel moves across an image/feature map to respond to local patterns such as edges, textures and shapes. |
| Feature map | An intermediate spatial representation showing where learned features respond. Detection heads use multi-scale features for objects of different sizes. |
| Stride | How far an operation moves or downsamples. Larger effective stride reduces spatial resolution and can hurt small-object localization. |
| Padding | Adds pixels around feature-map borders so convolution can control output size and retain edge information. |
| Receptive field | The image region influencing one feature. Larger context helps identify an object, but precise localization also needs fine spatial detail. |
| Classification | Predicts a class for an image or crop. |
| Detection | Predicts class plus bounding box for potentially multiple objects; this project uses detection. |
| Segmentation | Predicts pixel-level regions; not implemented here. |
| Bounding-box regression | Model prediction of box coordinates relative to the object. |
| Confidence | Model score for a detection; threshold is 0.25 here and is not guaranteed to be calibrated probability. |
| IoU | Intersection divided by union. Used for NMS, tracking and detection scoring. |
| NMS | Removes overlapping duplicate boxes. This project applies it after semantic mapping. |
| Precision | Fraction of predictions that are correct. |
| Recall | Fraction of ground-truth objects/events found. |
| F1 | Harmonic mean of precision and recall. |
| Tracking | Connects frame-level detections into persistent identities. |
| Data association | Decides which current detection belongs to which previous track. This project uses class-aware greedy IoU. |
| Temporal smoothing | Requires consistent evidence over time to avoid noisy one-frame events or OCR readings. |
| OCR | Converts text pixels into characters/numbers; used for the magazine counter. |
| HSV | Hue-saturation-value color space, useful for thresholding the yellow marker. |
| Contour | Boundary of a connected region in a binary mask; area, perimeter and moments support marker filtering/localization. |

---

# 11. Rapid-Fire Questions

1. **What is IoU?** Intersection area divided by union area; it measures box overlap from 0 to 1.
2. **Why tracking?** To preserve identity across frames and avoid treating each detection as a new event.
3. **Why `min_confirmation_frames`?** To reject short one-frame detections before emitting lifecycle events.
4. **What is its value?** Five consecutive frames.
5. **What is `disappearance_frames`?** Fifteen missing frames before a confirmed track emits disappearance.
6. **What is `max_missing_frames`?** Tracker retention, configured as thirty frames.
7. **Why must max missing be at least disappearance?** The ID must survive long enough for the event engine to announce disappearance.
8. **Why YOLO11n?** A small pretrained detector that provided a practical baseline for person/car/truck without custom training.
9. **Why not `enemy`?** A person detector cannot infer team affiliation reliably.
10. **Why merge car and truck?** YOLO alternated those labels for the same UAZ.
11. **What is semantic NMS?** NMS performed after source classes are mapped to semantic classes.
12. **Confidence threshold?** `0.25`.
13. **Semantic NMS threshold?** `0.3`.
14. **Tracker IoU threshold?** `0.1`.
15. **Evaluation IoU threshold?** `0.5`.
16. **Precision versus recall?** Precision measures false-alarm control; recall measures missed-object control.
17. **Overall detection F1?** About `0.9760`.
18. **Overall event F1?** `0.5`.
19. **Why the difference?** Tracking fragmentation, lifecycle logic and OCR misses accumulate after frame-level detection.
20. **Why HSV?** It expresses a color range more directly than raw BGR/RGB under moderate brightness variation.
21. **Are minimap positions world coordinates?** No, they are normalized coordinates relative to the minimap ROI.
22. **Does minimap go to JSON?** No, it is currently separate analysis/demo output.
23. **Why `ammo_changed`?** The observation does not uniquely prove a shot.
24. **How many OCR confirmations?** Three matching sampled observations.
25. **What is the biggest generalization limitation?** One clip with clip-specific windows, thresholds and ROIs.

---

# 12. Five Strongest Technical Decisions

1. Separating detection, tracking and temporal event reasoning.
2. Mapping `car` and `truck` before semantic NMS to stabilize the UAZ category.
3. Refusing to label generic person detections as enemies.
4. Resetting temporal state across montage segment boundaries.
5. Reporting weak event results and concrete failure cases instead of hiding them behind high detection F1.
