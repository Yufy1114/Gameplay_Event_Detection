"""Game-agnostic minimap ROI and marker localization primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import math

from .ocr import normalized_roi_to_pixels


@dataclass(frozen=True, slots=True)
class MinimapObject:
    object_type: str
    minimap_x: float
    minimap_y: float
    confidence: float

    def __post_init__(self) -> None:
        if not 0 <= self.minimap_x <= 1 or not 0 <= self.minimap_y <= 1:
            raise ValueError("Minimap coordinates must be normalized to [0, 1]")


class MinimapMarkerDetector(ABC):
    @abstractmethod
    def detect(self, minimap_crop: Any) -> list[tuple[str, float, float, float]]:
        """Return marker type, pixel center x/y, and confidence within the crop."""


class HSVMarkerDetector(MinimapMarkerDetector):
    """Conservative detector for a configured colored, approximately circular marker."""

    def __init__(
        self,
        hsv_lower: tuple[int, int, int],
        hsv_upper: tuple[int, int, int],
        min_area: int,
        max_area: int,
        min_circularity: float,
        border_min_pixels: int,
    ) -> None:
        import cv2
        import numpy as np

        self.cv2 = cv2
        self.np = np
        self.lower = np.array(hsv_lower, dtype=np.uint8)
        self.upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.border_min_pixels = border_min_pixels

    def detect(self, minimap_crop: Any) -> list[tuple[str, float, float, float]]:
        hsv = self.cv2.cvtColor(minimap_crop, self.cv2.COLOR_BGR2HSV)
        mask = self.cv2.inRange(hsv, self.lower, self.upper)
        border_width = max(1, round(mask.shape[1] * 0.09))
        if self.border_min_pixels and self.cv2.countNonZero(mask[:, -border_width:]) < self.border_min_pixels:
            return []
        contours, _ = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, float, float, float]] = []
        for contour in contours:
            area = float(self.cv2.contourArea(contour))
            perimeter = float(self.cv2.arcLength(contour, True))
            if not self.min_area <= area <= self.max_area or perimeter <= 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue
            moments = self.cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
            confidence = min(1.0, circularity)
            candidates.append((area, center_x, center_y, confidence))
        if not candidates:
            return []
        _, center_x, center_y, confidence = max(candidates)
        return [("map_marker", center_x, center_y, confidence)]


class MinimapAnalyzer:
    def __init__(
        self,
        roi: tuple[float, float, float, float],
        marker_detector: MinimapMarkerDetector | None = None,
    ) -> None:
        self.roi = roi
        self.marker_detector = marker_detector

    def analyze(self, frame: Any) -> list[MinimapObject]:
        if self.marker_detector is None:
            return []
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = normalized_roi_to_pixels(self.roi, width, height)
        crop = frame[y1:y2, x1:x2]
        crop_height, crop_width = crop.shape[:2]
        if crop_width == 0 or crop_height == 0:
            return []
        return [
            MinimapObject(
                object_type=object_type,
                minimap_x=max(0.0, min(1.0, center_x / crop_width)),
                minimap_y=max(0.0, min(1.0, center_y / crop_height)),
                confidence=confidence,
            )
            for object_type, center_x, center_y, confidence in self.marker_detector.detect(crop)
        ]
