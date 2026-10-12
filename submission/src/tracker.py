"""Tracker interface and lightweight class-aware IoU tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

from .types import BBox, Detection, TrackedObject


def calculate_iou(first: BBox, second: BBox) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], frame_id: int) -> list[TrackedObject]:
        """Associate current-frame detections with persistent track IDs."""

    @abstractmethod
    def retire(self, track_ids: set[int]) -> None:
        """End tracks whose event lifecycle has been declared disappeared."""


@dataclass(slots=True)
class _Track:
    track_id: int
    class_name: str
    bbox: BBox
    last_seen_frame: int


class IoUTracker(Tracker):
    """Simple greedy tracker suitable as a transparent baseline."""

    def __init__(self, iou_threshold: float = 0.3, max_missing_frames: int = 10) -> None:
        if not isinstance(iou_threshold, (int, float)) or isinstance(iou_threshold, bool) or not math.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be a finite number in [0, 1]")
        if not isinstance(max_missing_frames, int) or isinstance(max_missing_frames, bool) or max_missing_frames < 1:
            raise ValueError("max_missing_frames must be an integer >= 1")
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1

    def update(self, detections: list[Detection], frame_id: int) -> list[TrackedObject]:
        self._expire_tracks(frame_id)
        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                if track.class_name != detection.class_name:
                    continue
                score = calculate_iou(detection.bbox, track.bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, detection_index, track_id))

        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()
        assignments: dict[int, int] = {}
        for _, detection_index, track_id in sorted(candidates, reverse=True):
            if detection_index in matched_detections or track_id in matched_tracks:
                continue
            matched_detections.add(detection_index)
            matched_tracks.add(track_id)
            assignments[detection_index] = track_id

        tracked_objects: list[TrackedObject] = []
        for index, detection in enumerate(detections):
            track_id = assignments.get(index)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
            self._tracks[track_id] = _Track(
                track_id=track_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
                last_seen_frame=frame_id,
            )
            tracked_objects.append(TrackedObject.from_detection(detection, track_id))
        return tracked_objects

    def _expire_tracks(self, frame_id: int) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_id - track.last_seen_frame > self.max_missing_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]

    def retire(self, track_ids: set[int]) -> None:
        for track_id in track_ids:
            self._tracks.pop(track_id, None)
