"""Annotated video rendering for tracks and recent events."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from .types import GameplayEvent, TrackedObject


class VideoVisualizer:
    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        show_recent_events: bool = True,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Visualization requires OpenCV") from exc
        self.cv2 = cv2
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise OSError(f"Could not open annotated video output: {output_path}")
        self.show_recent_events = show_recent_events
        self.recent_events: deque[GameplayEvent] = deque(maxlen=5)

    def write(
        self,
        frame: Any,
        tracked_objects: list[TrackedObject],
        new_events: list[GameplayEvent],
    ) -> None:
        for tracked in tracked_objects:
            x1, y1, x2, y2 = (int(value) for value in tracked.bbox)
            self.cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            label = f"{tracked.class_name} | ID {tracked.track_id} | {tracked.confidence:.2f}"
            self.cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 7)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0),
                2,
                self.cv2.LINE_AA,
            )
        self.recent_events.extend(new_events)
        if self.show_recent_events:
            for index, event in enumerate(reversed(self.recent_events)):
                text = f"{event.timestamp:.2f}s {event.event}"
                self.cv2.putText(
                    frame,
                    text,
                    (12, 25 + index * 22),
                    self.cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (40, 220, 255),
                    2,
                    self.cv2.LINE_AA,
                )
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()

    def __enter__(self) -> "VideoVisualizer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
