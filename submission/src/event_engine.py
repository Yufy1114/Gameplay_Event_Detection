"""Stateful derivation of deduplicated object lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass

from .types import BBox, GameplayEvent, TrackedObject


@dataclass(slots=True)
class _ObjectState:
    class_name: str
    consecutive_seen: int
    last_seen_frame: int
    confidence: float
    bbox: BBox
    confirmed: bool = False


class EventEngine:
    def __init__(self, min_confirmation_frames: int = 3, disappearance_frames: int = 6) -> None:
        if (
            not isinstance(min_confirmation_frames, int)
            or isinstance(min_confirmation_frames, bool)
            or min_confirmation_frames < 1
            or not isinstance(disappearance_frames, int)
            or isinstance(disappearance_frames, bool)
            or disappearance_frames < 1
        ):
            raise ValueError("Temporal thresholds must be integers >= 1")
        self.min_confirmation_frames = min_confirmation_frames
        self.disappearance_frames = disappearance_frames
        self._states: dict[int, _ObjectState] = {}

    def update(
        self,
        tracked_objects: list[TrackedObject],
        frame_id: int,
        timestamp: float,
    ) -> list[GameplayEvent]:
        events: list[GameplayEvent] = []
        seen_ids: set[int] = set()

        for tracked in tracked_objects:
            seen_ids.add(tracked.track_id)
            state = self._states.get(tracked.track_id)
            if state is None or state.class_name != tracked.class_name:
                state = _ObjectState(
                    class_name=tracked.class_name,
                    consecutive_seen=1,
                    last_seen_frame=frame_id,
                    confidence=tracked.confidence,
                    bbox=tracked.bbox,
                )
                self._states[tracked.track_id] = state
            else:
                state.consecutive_seen = (
                    state.consecutive_seen + 1 if frame_id == state.last_seen_frame + 1 else 1
                )
                state.last_seen_frame = frame_id
                state.confidence = tracked.confidence
                state.bbox = tracked.bbox

            if not state.confirmed and state.consecutive_seen >= self.min_confirmation_frames:
                state.confirmed = True
                events.append(self._event(tracked.track_id, state, frame_id, timestamp, "detected"))

        disappeared_ids: list[int] = []
        for track_id, state in self._states.items():
            if track_id in seen_ids:
                continue
            missing_frames = frame_id - state.last_seen_frame
            if missing_frames >= self.disappearance_frames:
                if state.confirmed:
                    events.append(self._event(track_id, state, frame_id, timestamp, "disappeared"))
                disappeared_ids.append(track_id)
        for track_id in disappeared_ids:
            del self._states[track_id]
        return events

    @staticmethod
    def _event(
        track_id: int,
        state: _ObjectState,
        frame_id: int,
        timestamp: float,
        suffix: str,
    ) -> GameplayEvent:
        return GameplayEvent(
            timestamp=timestamp,
            frame_id=frame_id,
            event=f"{state.class_name}_{suffix}",
            object_id=track_id,
            object_type=state.class_name,
            confidence=state.confidence,
            bbox=state.bbox,
        )
