"""Detector interface and an Ultralytics YOLO implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import Detection
from .config import ClassFilterConfig


class DetectorError(RuntimeError):
    """Raised when a detector cannot be initialized or invoked."""


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: Any, frame_id: int, timestamp: float) -> list[Detection]:
        """Detect configured object classes in one BGR frame."""


class UltralyticsDetector(Detector):
    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.25,
        image_size: int = 960,
        target_classes: list[str] | None = None,
        class_mapping: dict[str, str] | None = None,
        class_filters: dict[str, ClassFilterConfig] | None = None,
        semantic_nms_iou_threshold: float | None = 0.6,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "Ultralytics is unavailable. Install requirements.txt before running detection."
            ) from exc
        try:
            self.model = YOLO(model_name)
        except Exception as exc:
            raise DetectorError(f"Could not load detector model '{model_name}': {exc}") from exc
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.target_classes = set(target_classes or [])
        self.class_mapping = class_mapping or {}
        self.class_filters = class_filters or {}
        self.semantic_nms_iou_threshold = semantic_nms_iou_threshold
        self.device = device

        model_names = self.model.names
        available = {
            str(name) for name in (model_names.values() if isinstance(model_names, dict) else model_names)
        }
        unknown = self.target_classes - available
        if unknown:
            raise DetectorError(
                f"Target classes not present in model: {sorted(unknown)}. "
                f"Available classes: {sorted(available)}"
            )

    def detect(self, frame: Any, frame_id: int, timestamp: float) -> list[Detection]:
        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(f"Detector inference failed at frame {frame_id}: {exc}") from exc
        if not results:
            return []

        try:
            result = results[0]
            names = result.names
            detections: list[Detection] = []
            if result.boxes is None:
                return detections
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = str(names[class_id])
                confidence = float(box.conf.item())
                if self.target_classes and class_name not in self.target_classes:
                    continue
                coords = tuple(float(value) for value in box.xyxy[0].tolist())
                semantic_name = self.class_mapping.get(class_name, class_name)
                if not self._passes_filter(semantic_name, coords, frame.shape):
                    continue
                detections.append(
                    Detection(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        class_name=semantic_name,
                        confidence=confidence,
                        bbox=coords,
                    )
                )
            return self._semantic_nms(detections)
        except (IndexError, KeyError, TypeError, ValueError, AttributeError) as exc:
            raise DetectorError(f"Invalid detector output at frame {frame_id}: {exc}") from exc

    def _passes_filter(self, class_name: str, bbox: tuple[float, ...], shape: Any) -> bool:
        class_filter = self.class_filters.get(class_name)
        if class_filter is None:
            return True
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            return False
        x1, y1, x2, y2 = bbox
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        area_ratio = box_width * box_height / (width * height)
        aspect_ratio = box_width / box_height if box_height > 0 else float("inf")
        return (
            class_filter.min_bbox_area_ratio <= area_ratio <= class_filter.max_bbox_area_ratio
            and aspect_ratio >= class_filter.min_aspect_ratio
            and (
                class_filter.max_aspect_ratio is None
                or aspect_ratio <= class_filter.max_aspect_ratio
            )
        )

    def _semantic_nms(self, detections: list[Detection]) -> list[Detection]:
        if self.semantic_nms_iou_threshold is None:
            return detections
        kept: list[Detection] = []
        for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if any(
                existing.class_name == candidate.class_name
                and _iou(existing.bbox, candidate.bbox) > self.semantic_nms_iou_threshold
                for existing in kept
            ):
                continue
            kept.append(candidate)
        return kept


def _iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0
