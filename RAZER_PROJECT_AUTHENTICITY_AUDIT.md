# Razer Computer Vision Intern — Project Authenticity Interview Audit

> 本报告严格基于当前仓库中的 README、配置、源码、标注、评估输出、实验记录和测试。无法由仓库证明的内容统一标记为 **NOT VERIFIED**。

## 目录

1. [项目事实表](#1-项目事实表)
2. [真实性追问](#2-真实性追问)
3. [项目可延伸的 CV 基础](#3-项目可延伸的-cv-基础)
4. [必须记住的数字](#4-必须记住的数字)
5. [危险问题与安全回答](#5-危险问题与安全回答)
6. [最可能暴露真实理解程度的 Top 30](#6-最可能暴露真实理解程度的-top-30)
7. [审计结论](#7-审计结论)

---

# 1. 项目事实表

## 1.1 数据

### 视频

当前工作区存在 3 个视频，但正式报告指标只对应 `gameplay_classic.webm`。

| 文件 | 字节数 | 分辨率 | OpenCV FPS | OpenCV 帧数 | 估算时长 | 有正式指标 |
|---|---:|---:|---:|---:|---:|---|
| `gameplay_classic.webm` | 57,574,789 | 1920×1080 | 30.0023175 | 8,292 | 276.38 s | 是 |
| `gameplay.webm` | 55,784,290 | 1920×1080 | 25.0 | 2,323 | 92.92 s | 否 |
| `gameplay_bentley.webm` | 8,631,268 | 2436×1124 | 30.001 | 889 | 29.63 s | 否 |

README 对正式视频记录的是 8,290 个 decoded frames、约 276.31 秒；当前 OpenCV 容器元数据报告 8,292 帧、约 276.38 秒。差异的精确原因是 **NOT VERIFIED**，可能与容器声明帧数和实际解码帧数不同有关。

- 正式视频 SHA-1：`e9768bd3b90d342abfd945006eca2fea32994a6e`
- 训练视频：仓库未记录
- 正式 train/validation/test split：**NOT VERIFIED / 不存在**
- 本项目把标注集描述为 targeted validation，不是 blind test benchmark。

### 分析窗口

- Vehicle window：37.0–42.5 s，共 5.5 s
- Character/ammo window：79.0–86.0 s，共 7.0 s
- 独立 minimap window：约 114.0–121.5 s，共 7.5 s

注意：`analysis_segments[].purpose` 只用于描述和日志，不会动态控制 YOLO 只运行某个类别。

### 检测标注

- 唯一标注帧：55
- GT 实例：167
- 语义类别：2
- Character：142 个，分布于 32 帧
- Vehicle：25 个，分布于 25 帧
- 两帧同时包含两个类别，因此类别帧数相加为 57，但唯一帧数为 55。
- 类别占比：character 约 85%，vehicle 约 15%，明显不均衡。

Ground truth 生成方式：预测辅助生成初始框，然后人工检查和修正每个采样帧及框，并人工补充 detector misses。因此属于 **prediction-assisted + manually reviewed**，不是纯自动，也不是从空白开始的盲标。

检测帧来自选定连续窗口，而非整段视频每一帧。仓库没有明确记录一个全局统一的采样间隔，因此精确采样规则为 **NOT VERIFIED**。

### 事件数据

Scored GT 共 8 条：

| 事件 | 数量 |
|---|---:|
| `character_detected` | 1 |
| `character_disappeared` | 1 |
| `ammo_changed` | 6 |
| scored `vehicle_detected` | 0 |

另有一个 36.5 秒的 `vehicle_detected` GT 被保留在边界审计中，但因 montage cut 不参与评分。

原始 `outputs/events.json` 有 14 条事件：8 个 character detected、2 个 character disappeared、1 个 vehicle detected、3 个 ammo changed。评估前排除了 6 个 segment-initialization predictions。

### OCR 实验数据

- Crop preprocessing experiment：29 个时间点，3 个变体，共 87 条记录
- README 报告：24/29 正确，即 82.8%
- 但 JSON 中 `raw_3x`、`gray_3x`、`otsu_3x` 单独精确匹配分别为 7/29、5/29、2/29。
- 因此 README 的 24/29 对应哪个最终设置，无法仅从该 JSON 复现：**NOT VERIFIED**。

Sequential OCR：

- sampled observations：43
- parseable：36
- parseable 且 confidence ≥ 0.25：34
- 稳定输出事件：3
- 实际稳定变化：14→12、12→10、10→8

### Minimap 数据

- positive：76，全部检测到 marker
- character negative：71，检测为 0
- inventory negative：51，检测为 0
- vehicle negative：56，检测为 0
- negative 总数：178，false positive 为 0

该结果只适用于选定样本，不是通用准确率。

## 1.2 模型

- 框架：Ultralytics
- 模型：YOLO11n
- 权重：`yolo11n.pt`
- 本地权重 SHA-1：`e14fc9da464f514a6fd44e4274ddcf295826c26b`
- 推理输入尺寸：960
- 使用预训练模型，没有仓库证据表明进行过 fine-tuning
- optimizer、learning rate、epochs、training loss：**NOT VERIFIED**
- custom training checkpoint：**NOT VERIFIED**

Raw target classes：`person`, `car`, `truck`。

Semantic mapping：

```text
person → character
car    → vehicle
truck  → vehicle
```

不命名为 enemy，因为通用 person detector 不能可靠识别阵营。合并 car/truck，是因为 README 记录同一 UAZ 在相邻帧可能被 YOLO 交替预测成二者。

每个 `Detection` 包含 frame ID、timestamp、映射后的 class、confidence 和 `(x1,y1,x2,y2)` box。完整 YOLO11n 内部逐层 backbone/neck/head 配置并未在项目报告中展开，因此精确层结构为 **NOT VERIFIED**。安全说法是：项目通过 Ultralytics 公共接口使用 YOLO11n，并没有修改内部网络结构。

YOLO11n 看起来是因为 nano 版本较轻、存在可用预训练类别、适合端到端原型。仓库中没有 YOLO11n/s/m/l 对照实验，因此“它是最优模型”是 **NOT VERIFIED**。

## 1.3 推理参数与后处理

| 参数 | 值 |
|---|---:|
| YOLO confidence | 0.25 |
| project semantic NMS IoU | 0.3 |
| tracker association IoU | 0.1 |
| tracker max missing | 30 frames |
| event confirmation | 5 consecutive frames |
| disappearance | 15 missing frames |
| detection evaluation IoU | 0.5 |
| event tolerance | ±0.5 s |

Geometry filters：

| 类别 | area ratio | aspect ratio |
|---|---|---|
| Character | 0.003–0.35 | 0.2–1.5 |
| Vehicle | 0.002–0.2 | 0.7–3.5 |

Semantic NMS 在 mapping 和 geometry filter 后执行。它按 confidence 降序，用相同 semantic class 且 IoU > 0.3 的条件抑制候选。

Tracker 是 class-aware、IoU-only、greedy、一对一的 tracking-by-detection baseline。没有 Kalman filter、appearance embedding 或 Hungarian assignment。

## 1.4 OCR 与 Minimap 参数

OCR：

- backend：EasyOCR
- reader language：English
- EasyOCR GPU：false
- normalized ROI：`[0.3583333333,0.9490740741,0.3723958333,0.9888888889]`
- 1920×1080 下约为 x=688:715、y=1025:1068
- 每 5 帧采样
- 4× bicubic resize
- digits-only allowlist
- greedy decoder
- minimum confidence 0.25
- 3 个一致观察才确认
- 首个稳定值只初始化，不产生 event

Minimap：

- main config 中 `enabled=false`
- ROI：`[0.7109375,0.0,0.8619791667,0.2777777778]`
- HSV lower `(22,120,120)`，upper `(38,255,255)`
- contour area 900–2200
- circularity ≥ 0.45
- 选择面积最大的有效 contour
- confidence 是 `min(1,circularity)`，不是 learned probability
- 返回 ROI-relative normalized coordinates，不是 world coordinates
- `run_pipeline` 调用了 analyzer 后没有把返回值写入事件流；demo 单独使用结果。

## 1.5 硬件与性能

- AMD Ryzen 9 7945HX
- 16 cores / 32 logical CPUs
- WSL2
- CPU-only，GPU unavailable
- decoded/read frames：2,581
- detector-analyzed frames：377（166 vehicle + 211 character）
- wall time：38.52 s
- throughput：9.79 analyzed frames/s
- peak RSS：2,236,916 KB，约 2.13 GiB
- 不包含首次模型下载和 demo rendering

不能声称达到源视频 30 FPS 实时处理。

## 1.6 评估

Detection matching：同一帧、同一 semantic class、IoU ≥ 0.5，候选按 IoU 从高到低 greedy one-to-one 匹配。未匹配 prediction 是 FP，未匹配 GT 是 FN。它是确定性的，但不保证所有构型下全局最优。

| Class | Frames | GT | Pred. | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Character | 32 | 142 | 143 | 139 | 4 | 3 | 0.972028 | 0.978873 | 0.975439 |
| Vehicle | 25 | 25 | 24 | 24 | 0 | 1 | 1.000000 | 0.960000 | 0.979592 |
| Overall | 55 | 167 | 167 | 163 | 4 | 4 | 0.976048 | 0.976048 | 0.976048 |

AP、AP50、AP75、mAP、COCO-style mAP：全部 **NOT VERIFIED**。0.976 是单一 operating point 下的 F1，不是 mAP。

Event matching：同 event type、时间差 ≤ 0.5 s，按最小时间差 greedy one-to-one 匹配。它不检查 object ID、bbox、ammo previous value 或 current value。

Initialization exclusion window：`5 / 30.0023175 = 0.1666537927 s`。共排除 6 个预测；另保留 1 个 montage-boundary GT 在审计文件中。

| Event | Pred. | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `character_detected` | 3 | 1 | 1 | 2 | 0 | 0.3333 | 1.0000 | 0.5000 |
| `character_disappeared` | 2 | 1 | 1 | 1 | 0 | 0.5000 | 1.0000 | 0.6667 |
| `vehicle_detected` | 0 | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| `ammo_changed` | 3 | 6 | 2 | 1 | 4 | 0.6667 | 0.3333 | 0.4444 |
| Overall | 8 | 8 | 4 | 4 | 4 | 0.5000 | 0.5000 | 0.5000 |

Frame-level detection F1 0.976，但 downstream event F1 只有 0.5。原因是 event 还受到 ID fragmentation、temporal state 和 OCR smoothing 的影响；高质量抽样帧检测并不自动代表完整时序事件准确。

## 1.7 失败案例

Character 主要错误来自 controlled avatar 的 nested/coarse boxes。高 confidence 粗框可能在 semantic NMS 中保留并抑制紧框，最后粗预测在 IoU 0.5 下成为 FP，而紧 GT 未匹配成为 FN。典型帧：2370、2384、2542。2528 也有 duplicate FP。

Vehicle 的唯一 FN 在 frame 1123：可见 UAZ 只有一个 oversized scope-overlapping `truck` prediction，随后被 vehicle maximum area filter 拒绝。这是 detector candidate 不佳和 postprocessing filter 共同导致。

最弱 downstream task 是 `ammo_changed`：recall 0.3333、F1 0.4444，六个 GT 中漏掉四个。三次一致观察提升稳定性，但跳过了短暂存在的 13、11、9。

## 1.8 测试状态

- 仓库定义 31 个 test methods。
- 当前只读审计运行：22 passed，9 errors。
- 9 个错误都发生在 `tempfile.TemporaryDirectory` 无法找到可写临时目录，未进入对应 assertion。
- 因此当前环境中的完整 31/31 passing 是 **NOT VERIFIED**；同时这些也不能描述成算法 assertion failures。

---

# 2. 真实性追问

下面每组按 Level 1 事实、Level 2 机制、Level 3 实验有效性展开。

## 2.1 数据规模

**L1：How much data did you actually use?**

> The documented metrics use one clip, 55 annotated frames, 167 detection GT instances and eight scored GT events.

**L2：How were the frames selected?**

> They came from selected continuous windows rather than the complete montage. A universal sampling interval is not formally documented.

**L3：Can this prove generalization?**

> No. It is targeted validation on one clip, not a broad held-out benchmark.

## 2.2 Ground truth

**L1：How was GT created?**

> Prediction-assisted boxes were manually reviewed and corrected, and misses were manually added.

**L2：What bias can that introduce?**

> Confirmation bias toward model-proposed boxes and possible overlooked omissions.

**L3：Why is the result still useful?**

> It validates the pipeline on a targeted reviewed set, but it should not be described as blind independent testing.

## 2.3 指标

**L1：What were TP, FP and FN?**

> 163 TP, 4 FP and 4 FN overall.

**L2：How did you calculate recall?**

> `163 / (163 + 4) = 0.976048`.

**L3：What happens if IoU changes to 0.75?**

> Recall can only stay the same or decrease for fixed predictions, especially because coarse boxes will fail. Exact values are NOT VERIFIED.

## 2.4 阈值

**L1：Name your IoU-related thresholds.**

> Semantic NMS 0.3, tracker association 0.1, evaluation 0.5.

**L2：Why are they different?**

> They control same-frame duplicate suppression, cross-frame association and scoring respectively.

**L3：What formal sweep selected them?**

> NOT VERIFIED. The repository does not contain a held-out threshold ablation.

## 2.5 Semantic NMS

**L1：Why semantic?**

> Mapping happens first, so car and truck both become vehicle and can suppress each other.

**L2：How is it implemented?**

> Confidence-sorted greedy suppression for same-semantic-class boxes with IoU greater than 0.3.

**L3：Did it hurt?**

> Yes. It sometimes kept a coarse controlled-character box and suppressed a tighter localization.

## 2.6 Tracking

**L1：How do you associate objects?**

> Same-class IoU candidates at least 0.1, sorted descending, then greedy one-to-one assignment.

**L2：Is it globally optimal?**

> No. It is not Hungarian matching and has no motion or appearance model.

**L3：What happens when characters cross?**

> ID switches or fragmentation can occur, creating duplicate lifecycle events.

## 2.7 ByteTrack

**L1：Why not ByteTrack?**

> The README comparison reports 1 vehicle and 8 character IDs for the IoU tracker versus 2 and 15 for ByteTrack in selected windows.

**L2：Does that prove IoU is superior?**

> No, only that the tested baseline was more stable in this limited case.

**L3：Where is the raw ByteTrack log?**

> NOT VERIFIED; the comparison exists in README but no dedicated raw script/log is present.

## 2.8 Event engine

**L1：When does detected fire?**

> On the fifth consecutive observation of an unconfirmed track.

**L2：When does disappeared fire?**

> After a confirmed track has been missing for 15 frames.

**L3：Are those physical event timestamps?**

> No. They are delayed confirmation timestamps.

## 2.9 FPS

**L1：What happens at 60 FPS?**

> Five frames changes from about 167 ms to 83 ms; fifteen changes from about 500 ms to 250 ms.

**L2：Why is that a problem?**

> The semantic duration changes even though the real-world behavior does not.

**L3：Production fix?**

> Express thresholds in seconds or use timestamp differences.

## 2.10 OCR

**L1：Describe the OCR pipeline.**

> Fixed normalized ROI, every-five-frame sampling, 4× resize, EasyOCR digit allowlist, confidence 0.25 and three matching observations.

**L2：Why was 13 skipped?**

> It appeared only twice before 12, so it was never confirmed.

**L3：Does smoothing improve everything?**

> No. It improves stability but adds latency and reduces recall.

## 2.11 Minimap

**L1：What coordinates are returned?**

> Normalized coordinates inside the minimap ROI.

**L2：Are they world coordinates?**

> No, there is no screen-to-world calibration.

**L3：Are they saved in events JSON?**

> No. The main pipeline discards the analyzer return value; the demo uses it separately.

## 2.12 Evaluation validity

**L1：How are ammo events matched?**

> Same event type within ±0.5 seconds.

**L2：Are previous/current ammo values checked?**

> No.

**L3：What validity problem does that create?**

> A temporally close but wrong ammo transition could count as a TP.

## 2.13 个人真实性问题

仓库不能证明：

- 哪些代码由候选人本人编写
- 最久的 bug 是什么
- 个人开发顺序
- 哪个实验最先失败
- 每个 commit 的作者
- 完成项目耗时

当前目录没有可用 Git history，因此以上均为 **NOT VERIFIED**。面试时必须根据真实经历回答，不能编故事。最好的真实性证据是能准确解释具体函数、状态转换、失败案例和取舍。

---

# 3. 项目可延伸的 CV 基础

## YOLO

```text
YOLO11n
→ classification vs detection
→ bbox regression
→ confidence
→ pretrained model and domain shift
→ small-object/localization error
→ IoU
→ NMS and semantic NMS
→ precision/recall/F1
→ AP/mAP vs single operating point
```

## Tracking

```text
tracking-by-detection
→ data association
→ class-aware matching
→ greedy vs Hungarian
→ IoU gating
→ track ID
→ temporary miss and retention
→ confirmation/disappearance
→ ID switches and fragmentation
→ Kalman filter / appearance embedding
→ ByteTrack and DeepSORT alternatives
```

当前实现没有 Kalman、appearance embedding、DeepSORT 或 Hungarian。

## Temporal reasoning

```text
observations
→ state machine
→ debouncing
→ consecutive confirmation
→ missing-frame threshold
→ deduplication
→ latency/stability trade-off
→ FPS dependence
→ montage boundary reset
```

## OCR

```text
HUD ROI
→ normalized crop
→ resize/interpolation
→ digit allowlist
→ OCR confidence
→ parsing
→ invalid-observation reset
→ temporal smoothing
→ skipped intermediate values
```

## Minimap

```text
BGR → HSV → threshold mask → external contours
→ area/perimeter → circularity → moments/centroid
→ ROI-relative normalized coordinate
```

圆度：$C=4\pi A/P^2$。质心：$c_x=m_{10}/m_{00}$，$c_y=m_{01}/m_{00}$。

## Evaluation

```text
GT creation
→ sampling bias
→ prediction-assisted annotation
→ TP/FP/FN
→ class-aware one-to-one matching
→ IoU threshold
→ precision/recall/F1
→ AP/mAP distinction
→ temporal tolerance
→ boundary exclusion
→ held-out validity
```

Segmentation、Mask R-CNN、FPN、RPN、RoIAlign、optical flow 和 VLM 不是当前项目组件，只能作为替代方案讨论。

---

# 4. 必须记住的数字

| Item | Exact value | Why asked | Evidence |
|---|---:|---|---|
| 正式视频大小 | 57,574,789 bytes | 可复现性 | README/file |
| SHA-1 | `e9768bd3b90d342abfd945006eca2fea32994a6e` | 输入确认 | README/file |
| 分辨率/FPS | 1920×1080 / 30.0023175 | ROI 和时序 | README/metrics |
| README 帧数/时长 | 8,290 / ~276.31 s | 数据规模 | README |
| Model | YOLO11n, `yolo11n.pt` | 模型选择 | config |
| Input size | 960 | 推理设置 | config |
| Confidence | 0.25 | P/R trade-off | config |
| Semantic NMS IoU | 0.3 | 同帧去重 | config |
| Tracker IoU | 0.1 | 跨帧关联 | config |
| Max missing | 30 frames | ID retention | config |
| Confirmation | 5 frames | detected event | config |
| Disappearance | 15 frames | disappeared event | config |
| Eval IoU | 0.5 | detection matching | config |
| Annotated frames | 55 | 数据规模 | annotations |
| GT objects | 167 | 数据规模 | metrics |
| Character GT | 142 | 分布 | metrics |
| Vehicle GT | 25 | 分布 | metrics |
| Character TP/FP/FN | 139/4/3 | 结果 | metrics |
| Character P/R/F1 | .972028/.978873/.975439 | 结果 | metrics |
| Vehicle TP/FP/FN | 24/0/1 | 结果 | metrics |
| Vehicle P/R/F1 | 1/.96/.979592 | 结果 | metrics |
| Overall TP/FP/FN | 163/4/4 | 最重要结果 | metrics |
| Overall P/R/F1 | .976048/.976048/.976048 | 最重要结果 | metrics |
| AP/mAP | **NOT VERIFIED** | 陷阱问题 | 无产物 |
| Event tolerance | ±0.5 s | 时间匹配 | config |
| Initialization window | .1666537927 s | 边界排除 | metrics |
| Excluded predictions | 6 | 审计 | exclusion JSON |
| Event GT/predictions | 8/8 | 事件规模 | metrics |
| Event TP/FP/FN | 4/4/4 | 最终系统 | metrics |
| Event P/R/F1 | .5/.5/.5 | 最终系统 | metrics |
| Ammo TP/FP/FN | 2/1/4 | 最弱任务 | metrics |
| Ammo P/R/F1 | .6667/.3333/.4444 | 最弱任务 | metrics |
| OCR interval/upscale | 5 frames / 4× | OCR 行为 | config |
| OCR confidence/stability | .25 / 3 observations | OCR smoothing | config |
| OCR samples | 43; 36 parseable; 34 passed | OCR 实验 | experiment |
| Minimap HSV H | 22–38 | 分割 | config |
| Marker area/circularity | 900–2200 / ≥.45 | 形状筛选 | config |
| Minimap result | 76/76 positive; 0/178 negative detections | 选定样本 | observations |
| Runtime | 38.52 s | 性能 | README |
| Throughput | 9.79 analyzed FPS | 性能 | README |
| Tests | 31 defined | 工程质量 | tests |

---

# 5. 危险问题与安全回答

## What was your mAP?

- Verified：IoU 0.5、confidence 0.25 下的 P/R/F1
- NOT VERIFIED：AP、AP50、AP75、mAP
- Safe：`I did not compute mAP. I report precision, recall and F1 at a fixed operating point.`
- 禁止：把 0.976 称为 mAP。

## How did you train YOLO?

- Verified：使用 pretrained YOLO11n inference
- NOT VERIFIED：fine-tuning、optimizer、LR、epochs
- Safe：`I did not train YOLO. My work was the adaptation, tracking, temporal reasoning and evaluation pipeline.`
- 禁止：声称 custom training。

## What was the train/validation/test split?

- Verified：55-frame targeted validation
- NOT VERIFIED：正式训练/验证/测试拆分
- Safe：`There was no detector training split. The annotated set is targeted validation, not a blind test set.`
- 禁止：编造 80/10/10。

## Why YOLO11n rather than YOLO11s?

- Verified：nano 版本被使用，适合轻量 baseline
- NOT VERIFIED：模型尺寸 ablation
- Safe：`I selected a small pretrained baseline, but did not establish that it was globally optimal.`

## What happened when thresholds changed?

- Verified：当前阈值和理论影响
- NOT VERIFIED：完整 sweep 的具体数字
- Safe：解释方向性影响，并明确 alternative metrics 未计算。

## Was OCR accuracy really 82.8%?

- Verified：README 报告 24/29
- NOT VERIFIED：从现有 preprocessing JSON 直接重建 24/29
- Safe：将其称为 limited reported feasibility check。

## Did all tests pass?

- Verified：31 tests defined；当前 22 pass，9 因无可写 temp dir 而 error
- NOT VERIFIED：当前环境完整 31/31
- Safe：明确环境错误发生在 assertion 之前。

## Did you benchmark ByteTrack?

- Verified：README 中记录 1/8 vs 2/15 IDs
- NOT VERIFIED：原始脚本、配置和 log
- Safe：称为 selected-window limited comparison，而非通用 benchmark。

## Is minimap accuracy 100%?

- Verified：选定样本 76/76，178 negatives 中 0 detections
- NOT VERIFIED：跨游戏泛化
- Safe：只陈述 selected samples。

## Who wrote the code / what bug took longest?

- Verified：最终文件
- NOT VERIFIED：个人历史，无 Git history
- Safe：根据真实经历回答，并用函数级细节证明理解。

---

# 6. 最可能暴露真实理解程度的 Top 30

| # | Question | Short answer | Likely follow-up | Follow-up answer | Evidence |
|---:|---|---|---|---|---|
| 1 | How much data? | One evaluated clip, 55 frames, 167 GT objects, 8 scored GT events. | Is it a test set? | Targeted validation, not blind test. | annotations/metrics |
| 2 | Exact detection result? | 163 TP, 4 FP, 4 FN; P/R/F1=.976048. | Why P=R? | GT and predictions both 167; FP=FN=4. | detection metrics |
| 3 | Class counts? | 142 character, 25 vehicle. | Balanced? | No, ~85% character. | annotations |
| 4 | GT creation? | Prediction-assisted, then manually reviewed/corrected; misses added. | Bias? | Confirmation bias risk. | README |
| 5 | Eval IoU? | 0.5. | At 0.75? | Recall same or lower; exact result NOT VERIFIED. | config/evaluator |
| 6 | mAP? | Not computed. | What is .976? | F1 at one operating point. | evaluator |
| 7 | IoU thresholds? | NMS .3, tracker .1, eval .5. | Why different? | Deduplication, association and scoring. | config |
| 8 | Semantic NMS? | Map car/truck to vehicle before NMS. | Failure? | Coarse character box suppressed tight box. | detector/failures |
| 9 | Why character not enemy? | Person model cannot infer team. | How add enemy? | Separate affiliation evidence/model. | mapping/README |
| 10 | Tracker algorithm? | Same-class greedy IoU one-to-one. | Global optimum? | No Hungarian. | tracker |
| 11 | ID switch cause? | Overlap, motion, jitter, occlusion, no appearance. | Event impact? | Extra lifecycle events. | tracker/event metrics |
| 12 | Why not ByteTrack? | README reports fewer IDs for IoU baseline on selected windows. | General superiority? | No. | README |
| 13 | Detected trigger? | Fifth consecutive observation. | Timestamp meaning? | Confirmation time, not first appearance. | event engine |
| 14 | Disappeared trigger? | Confirmed ID missing 15 frames. | Why retain 30? | Must outlive disappearance window. | engine/config |
| 15 | Missed for 10 frames? | No disappearance; ID may recover. | Five-frame reconfirm? | Not if already confirmed. | state logic |
| 16 | Why reset segments? | Montage cuts break continuity. | Automatic cut detection? | No, configured windows. | main/config |
| 17 | Why exclude six events? | Segment initialization within .166654 s. | Manipulation? | Audited, pre-match, type-agnostic, but still a protocol choice. | exclusion JSON |
| 18 | Detection F1 vs event F1? | Tracking and OCR errors accumulate downstream. | Weakest event? | Ammo F1 .4444. | metrics |
| 19 | Event matching? | Same type, ±.5 s, closest one-to-one. | Ammo values checked? | No. | evaluator |
| 20 | OCR pipeline? | ROI, every 5 frames, 4×, digits, .25, three matches. | Why skip 13? | Only two observations. | OCR log/code |
| 21 | Why ammo_changed? | Value change does not prove a shot. | Extra evidence? | Audio, flash, recoil, telemetry. | README |
| 22 | OCR result? | 43 samples, 36 parseable, 34 passed, 3 events. | Recall? | Ammo recall .3333. | OCR/event metrics |
| 23 | Minimap algorithm? | HSV, contour area, circularity, moments. | Confidence? | Circularity, not learned. | minimap |
| 24 | World coordinates? | No, ROI-relative only. | Saved to JSON? | No; demo-only/analysis channel. | main/demo |
| 25 | Vehicle FN? | Frame 1123 oversized truck/scope box filtered. | Remove filter? | Validate trade-off separately first. | failure cases |
| 26 | Character errors? | Nested/coarse controlled-avatar boxes. | One object both FP/FN? | Coarse prediction unmatched; tight GT unmatched. | failure cases |
| 27 | FPS doubles? | Frame thresholds represent half the time. | Fix? | Seconds/timestamps. | state code |
| 28 | Hard-coded/game-specific? | Windows, mappings, filters, ROIs, HSV, thresholds. | Reusable? | Interfaces/state/evaluation skeleton. | config/modules |
| 29 | Fine-tuning? | None; pretrained YOLO/EasyOCR. | Why not? | No large independent training set. | repo absence/code |
| 30 | Reproduce? | Exact video/SHA, dependencies, config, cached models, run module. | Bit-for-bit? | NOT VERIFIED; versions are ranges. | README/requirements |

---

# 7. 审计结论

最能证明候选人真实理解项目的，不是背诵 YOLO 定义，而是能准确说出以下实现细节：

- Mapping 在 project-level semantic NMS 之前。
- Semantic NMS 用 `IoU > 0.3`，tracker candidate 用 `IoU >= 0.1`。
- Tracker 自身过期条件是 missing `> 30`，event disappearance 是 `>= 15`。
- 未确认 track 中断后 consecutive count 会重置。
- 已确认 track 短暂丢失后不会重新产生 detected event。
- OCR 首个稳定值只初始化，不生成 event。
- 无效或低置信 OCR 会清空 candidate streak。
- OCR event confidence 取确认观察中的最小 confidence。
- Event evaluator 不检查 object ID 和 ammo values。
- Minimap confidence 是 circularity，不是模型概率。
- Minimap 返回值没有进入主 JSON/CSV 事件流。
- Segment `purpose` 不控制推理类别。
- 0.976 是 F1，不是 mAP。
- 当前项目没有训练 YOLO。
- 所有无法被仓库证明的个人经历和实验结果都应诚实标为 **NOT VERIFIED**，不要补充猜测。
