"""JSON configuration loading and validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when a pipeline configuration is invalid."""


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"'{name}' must be a JSON object")
    return value


def _roi(value: Any, field_name: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ConfigurationError(f"{field_name} must be [x1, y1, x2, y2]")
    if any(isinstance(v, bool) for v in value):
        raise ConfigurationError(f"{field_name} must contain numeric coordinates")
    try:
        result = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field_name} must contain numeric coordinates") from exc
    x1, y1, x2, y2 = result
    if not all(math.isfinite(v) for v in result) or not (
        0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
    ):
        raise ConfigurationError(f"{field_name} must use normalized coordinates in [0, 1]")
    return result


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


@dataclass(slots=True)
class DetectorConfig:
    backend: str = "ultralytics"
    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.25
    image_size: int = 960
    target_classes: list[str] = field(default_factory=list)
    class_mapping: dict[str, str] = field(default_factory=dict)
    class_filters: dict[str, "ClassFilterConfig"] = field(default_factory=dict)
    semantic_nms_iou_threshold: float | None = 0.6
    device: str | None = None


@dataclass(slots=True)
class ClassFilterConfig:
    min_bbox_area_ratio: float = 0.0
    max_bbox_area_ratio: float = 1.0
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class AnalysisSegment:
    start: float
    end: float
    purpose: str


@dataclass(slots=True)
class TrackerConfig:
    backend: str = "iou"
    iou_matching_threshold: float = 0.3
    max_missing_frames: int = 10


@dataclass(slots=True)
class TemporalConfig:
    min_confirmation_frames: int = 3
    disappearance_frames: int = 6


@dataclass(slots=True)
class OCRConfig:
    enabled: bool = False
    backend: str = "tesseract"
    rois: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    min_stable_frames: int = 2
    sample_every_frames: int = 1
    upscale_factor: int = 3
    min_confidence: float = 0.0


@dataclass(slots=True)
class MinimapConfig:
    enabled: bool = False
    roi: tuple[float, float, float, float] | None = None
    hsv_lower: tuple[int, int, int] = (22, 120, 120)
    hsv_upper: tuple[int, int, int] = (38, 255, 255)
    marker_min_area: int = 900
    marker_max_area: int = 2200
    marker_min_circularity: float = 0.6
    border_min_pixels: int = 500


@dataclass(slots=True)
class OutputConfig:
    json_path: str | None = "outputs/events.json"
    csv_path: str | None = None
    visualization_enabled: bool = False
    video_path: str | None = "outputs/annotated.mp4"
    show_recent_events: bool = True


@dataclass(slots=True)
class EvaluationConfig:
    detection_iou_threshold: float = 0.5
    event_time_tolerance_seconds: float = 0.5


@dataclass(slots=True)
class PipelineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    temporal_smoothing: TemporalConfig = field(default_factory=TemporalConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    minimap: MinimapConfig = field(default_factory=MinimapConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    analysis_segments: list[AnalysisSegment] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        config_path = Path(path)
        try:
            raw = json.loads(
                config_path.read_text(encoding="utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ConfigurationError(f"Non-finite JSON number is not allowed: {value}")
                ),
            )
        except OSError as exc:
            raise ConfigurationError(f"Cannot read configuration: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON in {config_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError("Configuration root must be a JSON object")

        detector = _section(raw, "detector")
        tracker = _section(raw, "tracker")
        temporal = _section(raw, "temporal_smoothing")
        ocr = _section(raw, "ocr")
        minimap = _section(raw, "minimap")
        output = _section(raw, "output")
        evaluation = _section(raw, "evaluation")
        raw_segments = raw.get("analysis_segments", [])
        if not isinstance(raw_segments, list):
            raise ConfigurationError("analysis_segments must be a JSON array")
        raw_mapping = detector.get("class_mapping", {})
        if not isinstance(raw_mapping, dict):
            raise ConfigurationError("detector.class_mapping must be a JSON object")
        raw_filters = detector.get("class_filters", {})
        if not isinstance(raw_filters, dict):
            raise ConfigurationError("detector.class_filters must be a JSON object")
        raw_rois = ocr.get("rois", {})
        if not isinstance(raw_rois, dict):
            raise ConfigurationError("ocr.rois must be a JSON object")
        try:
            instance = cls(
                detector=DetectorConfig(
                    **{
                        k: v
                        for k, v in detector.items()
                        if k not in {"class_mapping", "class_filters"}
                    },
                    class_mapping=raw_mapping,
                    class_filters={
                        name: ClassFilterConfig(**value)
                        for name, value in raw_filters.items()
                    },
                ),
                tracker=TrackerConfig(**tracker),
                temporal_smoothing=TemporalConfig(**temporal),
                ocr=OCRConfig(
                    **{k: v for k, v in ocr.items() if k != "rois"},
                    rois={name: _roi(value, f"ocr.rois.{name}") for name, value in raw_rois.items()},
                ),
                minimap=MinimapConfig(
                    **{
                        k: tuple(v) if k in {"hsv_lower", "hsv_upper"} else v
                        for k, v in minimap.items()
                        if k != "roi"
                    },
                    roi=_roi(minimap.get("roi"), "minimap.roi"),
                ),
                output=OutputConfig(**output),
                evaluation=EvaluationConfig(**evaluation),
                analysis_segments=[AnalysisSegment(**segment) for segment in raw_segments],
            )
        except TypeError as exc:
            raise ConfigurationError(f"Unknown or missing configuration field: {exc}") from exc
        instance.validate()
        return instance

    def validate(self) -> None:
        if self.detector.backend != "ultralytics":
            raise ConfigurationError("Supported detector backend: 'ultralytics'")
        if not isinstance(self.detector.model_name, str) or not self.detector.model_name.strip():
            raise ConfigurationError("detector.model_name cannot be empty")
        if not _is_finite_number(self.detector.confidence_threshold) or not (
            0 <= self.detector.confidence_threshold <= 1
        ):
            raise ConfigurationError("detector.confidence_threshold must be in [0, 1]")
        if not _is_positive_int(self.detector.image_size):
            raise ConfigurationError("detector.image_size must be an integer >= 1")
        if not isinstance(self.detector.target_classes, list) or not all(
            isinstance(name, str) and name for name in self.detector.target_classes
        ):
            raise ConfigurationError("detector.target_classes must contain non-empty strings")
        if not all(
            isinstance(source, str)
            and source
            and isinstance(target, str)
            and target
            for source, target in self.detector.class_mapping.items()
        ):
            raise ConfigurationError("detector.class_mapping must map non-empty strings")
        if self.detector.semantic_nms_iou_threshold is not None and (
            not _is_finite_number(self.detector.semantic_nms_iou_threshold)
            or not 0 <= self.detector.semantic_nms_iou_threshold <= 1
        ):
            raise ConfigurationError("detector.semantic_nms_iou_threshold must be in [0, 1] or null")
        unknown_mapping_sources = set(self.detector.class_mapping) - set(
            self.detector.target_classes
        )
        if self.detector.target_classes and unknown_mapping_sources:
            raise ConfigurationError(
                "detector.class_mapping sources must be present in detector.target_classes"
            )
        for class_name, filter_config in self.detector.class_filters.items():
            if not isinstance(class_name, str) or not class_name:
                raise ConfigurationError("detector.class_filters keys must be non-empty strings")
            numeric_values = (
                filter_config.min_bbox_area_ratio,
                filter_config.max_bbox_area_ratio,
                filter_config.min_aspect_ratio,
            )
            if not all(_is_finite_number(value) and value >= 0 for value in numeric_values):
                raise ConfigurationError(f"Invalid detector.class_filters.{class_name}")
            if not 0 <= filter_config.min_bbox_area_ratio <= filter_config.max_bbox_area_ratio <= 1:
                raise ConfigurationError(
                    f"Invalid bbox area range for detector.class_filters.{class_name}"
                )
            if filter_config.max_aspect_ratio is not None and (
                not _is_finite_number(filter_config.max_aspect_ratio)
                or filter_config.max_aspect_ratio < filter_config.min_aspect_ratio
            ):
                raise ConfigurationError(
                    f"Invalid aspect ratio range for detector.class_filters.{class_name}"
                )
        if self.detector.device is not None and not isinstance(self.detector.device, str):
            raise ConfigurationError("detector.device must be a string or null")
        if self.tracker.backend != "iou":
            raise ConfigurationError("Supported tracker backend: 'iou'")
        if not _is_finite_number(self.tracker.iou_matching_threshold) or not (
            0 <= self.tracker.iou_matching_threshold <= 1
        ):
            raise ConfigurationError("tracker.iou_matching_threshold must be in [0, 1]")
        if not _is_positive_int(self.tracker.max_missing_frames):
            raise ConfigurationError("tracker.max_missing_frames must be >= 1")
        if not _is_positive_int(self.temporal_smoothing.min_confirmation_frames):
            raise ConfigurationError("temporal_smoothing.min_confirmation_frames must be >= 1")
        if not _is_positive_int(self.temporal_smoothing.disappearance_frames):
            raise ConfigurationError("temporal_smoothing.disappearance_frames must be >= 1")
        if self.tracker.max_missing_frames < self.temporal_smoothing.disappearance_frames:
            raise ConfigurationError("tracker.max_missing_frames must be >= disappearance_frames")
        if self.ocr.backend not in {"tesseract", "easyocr"}:
            raise ConfigurationError("Supported OCR backends: 'tesseract', 'easyocr'")
        if not isinstance(self.ocr.enabled, bool):
            raise ConfigurationError("ocr.enabled must be a boolean")
        if not all(isinstance(name, str) and name for name in self.ocr.rois):
            raise ConfigurationError("ocr.rois keys must be non-empty strings")
        if self.ocr.enabled and not self.ocr.rois:
            raise ConfigurationError("ocr.rois must be configured when OCR is enabled")
        if not _is_positive_int(self.ocr.min_stable_frames):
            raise ConfigurationError("ocr.min_stable_frames must be >= 1")
        if not _is_positive_int(self.ocr.sample_every_frames):
            raise ConfigurationError("ocr.sample_every_frames must be >= 1")
        if not _is_positive_int(self.ocr.upscale_factor):
            raise ConfigurationError("ocr.upscale_factor must be >= 1")
        if not _is_finite_number(self.ocr.min_confidence) or not 0 <= self.ocr.min_confidence <= 1:
            raise ConfigurationError("ocr.min_confidence must be in [0, 1]")
        if self.minimap.enabled and self.minimap.roi is None:
            raise ConfigurationError("minimap.roi must be configured when minimap is enabled")
        if not isinstance(self.minimap.enabled, bool):
            raise ConfigurationError("minimap.enabled must be a boolean")
        for name, bound in (("hsv_lower", self.minimap.hsv_lower), ("hsv_upper", self.minimap.hsv_upper)):
            if len(bound) != 3 or not all(isinstance(v, int) and 0 <= v <= 255 for v in bound):
                raise ConfigurationError(f"minimap.{name} must contain three integers in [0, 255]")
        if not _is_positive_int(self.minimap.marker_min_area) or not _is_positive_int(self.minimap.marker_max_area) or self.minimap.marker_min_area > self.minimap.marker_max_area:
            raise ConfigurationError("minimap marker area range is invalid")
        if not _is_finite_number(self.minimap.marker_min_circularity) or not 0 <= self.minimap.marker_min_circularity <= 1:
            raise ConfigurationError("minimap.marker_min_circularity must be in [0, 1]")
        if (
            not isinstance(self.minimap.border_min_pixels, int)
            or isinstance(self.minimap.border_min_pixels, bool)
            or self.minimap.border_min_pixels < 0
        ):
            raise ConfigurationError("minimap.border_min_pixels must be >= 0")
        if not _is_finite_number(self.evaluation.detection_iou_threshold) or not (
            0 <= self.evaluation.detection_iou_threshold <= 1
        ):
            raise ConfigurationError("evaluation.detection_iou_threshold must be in [0, 1]")
        if not _is_finite_number(self.evaluation.event_time_tolerance_seconds) or (
            self.evaluation.event_time_tolerance_seconds < 0
        ):
            raise ConfigurationError("evaluation.event_time_tolerance_seconds must be >= 0")
        if self.output.visualization_enabled and not self.output.video_path:
            raise ConfigurationError("output.video_path is required when visualization is enabled")
        if not isinstance(self.output.visualization_enabled, bool):
            raise ConfigurationError("output.visualization_enabled must be a boolean")
        if not isinstance(self.output.show_recent_events, bool):
            raise ConfigurationError("output.show_recent_events must be a boolean")
        if not self.output.json_path and not self.output.csv_path:
            raise ConfigurationError("At least one JSON or CSV output path is required")
        for name, path in (("json_path", self.output.json_path), ("csv_path", self.output.csv_path)):
            if path is not None and (not isinstance(path, str) or not path.strip()):
                raise ConfigurationError(f"output.{name} must be a non-empty string or null")
        if self.output.video_path is not None and (
            not isinstance(self.output.video_path, str) or not self.output.video_path.strip()
        ):
            raise ConfigurationError("output.video_path must be a non-empty string or null")
        previous_end = -1.0
        for index, segment in enumerate(self.analysis_segments):
            if (
                not _is_finite_number(segment.start)
                or not _is_finite_number(segment.end)
                or segment.start < 0
                or segment.end <= segment.start
            ):
                raise ConfigurationError(f"Invalid analysis_segments[{index}] time range")
            if not isinstance(segment.purpose, str) or not segment.purpose.strip():
                raise ConfigurationError(f"analysis_segments[{index}].purpose cannot be empty")
            if segment.start < previous_end:
                raise ConfigurationError("analysis_segments must be sorted and non-overlapping")
            previous_end = segment.end
