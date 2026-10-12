"""Shared, serializable data types used across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, TypeAlias

BBox: TypeAlias = tuple[float, float, float, float]


def validate_bbox(bbox: BBox) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must contain [x1, y1, x2, y2]")
    if not all(isinstance(value, Real) and math.isfinite(float(value)) for value in bbox):
        raise ValueError("bbox coordinates must be finite numbers")
    x1, y1, x2, y2 = bbox
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox must satisfy x2 >= x1 and y2 >= y1")


@dataclass(frozen=True, slots=True)
class Detection:
    frame_id: int
    timestamp: float
    class_name: str
    confidence: float
    bbox: BBox

    def __post_init__(self) -> None:
        validate_bbox(self.bbox)
        if not isinstance(self.frame_id, Integral) or isinstance(self.frame_id, bool) or self.frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        if not isinstance(self.timestamp, Real) or not math.isfinite(float(self.timestamp)) or self.timestamp < 0:
            raise ValueError("timestamp must be a finite non-negative number")
        if not isinstance(self.class_name, str) or not self.class_name:
            raise ValueError("class_name cannot be empty")
        if not isinstance(self.confidence, Real) or not math.isfinite(float(self.confidence)) or not (
            0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TrackedObject:
    frame_id: int
    timestamp: float
    class_name: str
    confidence: float
    bbox: BBox
    track_id: int

    def __post_init__(self) -> None:
        validate_bbox(self.bbox)
        if not isinstance(self.class_name, str) or not self.class_name:
            raise ValueError("class_name cannot be empty")
        if not isinstance(self.track_id, Integral) or isinstance(self.track_id, bool) or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")
        if not isinstance(self.frame_id, Integral) or isinstance(self.frame_id, bool) or self.frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        if not isinstance(self.timestamp, Real) or not math.isfinite(float(self.timestamp)) or self.timestamp < 0:
            raise ValueError("timestamp must be a finite non-negative number")
        if not isinstance(self.confidence, Real) or not math.isfinite(float(self.confidence)) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

    @classmethod
    def from_detection(cls, detection: Detection, track_id: int) -> "TrackedObject":
        return cls(**asdict(detection), track_id=track_id)


@dataclass(frozen=True, slots=True)
class GameplayEvent:
    timestamp: float
    frame_id: int
    event: str
    object_id: int | str | None
    object_type: str | None
    confidence: float | None
    bbox: BBox | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, Integral) or isinstance(self.frame_id, bool) or self.frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        if not isinstance(self.timestamp, Real) or not math.isfinite(float(self.timestamp)) or self.timestamp < 0:
            raise ValueError("timestamp must be a finite non-negative number")
        if not isinstance(self.event, str) or not self.event:
            raise ValueError("event cannot be empty")
        if self.object_type is not None and not isinstance(self.object_type, str):
            raise ValueError("object_type must be a string or null")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary or null")
        if self.confidence is not None and (
            not isinstance(self.confidence, Real)
            or not math.isfinite(float(self.confidence))
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be between 0 and 1 or null")
        if self.bbox is not None:
            validate_bbox(self.bbox)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.bbox is not None:
            result["bbox"] = list(self.bbox)
        if self.metadata is None:
            result.pop("metadata")
        return _to_json_value(result)


def _to_json_value(value: Any) -> Any:
    """Recursively convert common NumPy-like values to JSON-native values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("Event output cannot contain NaN or Infinity")
        return converted
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_json_value(value.tolist())
    if hasattr(value, "item"):
        return _to_json_value(value.item())
    raise TypeError(f"Unsupported event output type: {type(value).__name__}")
