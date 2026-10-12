"""Optional, ROI-driven OCR with generic temporal value validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .types import GameplayEvent

ValueParser = Callable[[str], int | float | str | None]


def normalized_roi_to_pixels(
    roi: tuple[float, float, float, float], frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    return (
        int(round(x1 * frame_width)),
        int(round(y1 * frame_height)),
        int(round(x2 * frame_width)),
        int(round(y2 * frame_height)),
    )


def numeric_parser(text: str) -> int | float | None:
    """Parse a generic integer or decimal OCR value without UI-specific rules."""
    cleaned = "".join(char for char in text.strip() if char.isdigit() or char in ".-")
    if not cleaned or cleaned in {".", "-", "-."}:
        return None
    try:
        return float(cleaned) if "." in cleaned else int(cleaned)
    except ValueError:
        return None


@dataclass(slots=True)
class _ValueState:
    candidate: int | float | str | None = None
    candidate_frames: int = 0
    confirmed: int | float | str | None = None
    confidences: list[float] | None = None


class OCRProcessor:
    def __init__(
        self,
        rois: dict[str, tuple[float, float, float, float]],
        min_stable_frames: int = 2,
        parsers: dict[str, ValueParser] | None = None,
        backend: str = "tesseract",
        sample_every_frames: int = 1,
        upscale_factor: int = 3,
        min_confidence: float = 0.0,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OCR requires OpenCV") from exc
        self.cv2 = cv2
        self.backend = backend
        self.reader = None
        self.pytesseract = None
        if backend == "easyocr":
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError("EasyOCR backend requires the easyocr package") from exc
            self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        elif backend == "tesseract":
            try:
                import pytesseract
            except ImportError as exc:
                raise RuntimeError("Tesseract backend requires pytesseract") from exc
            self.pytesseract = pytesseract
        else:
            raise ValueError(f"Unsupported OCR backend: {backend}")
        self.rois = rois
        self.min_stable_frames = min_stable_frames
        self.parsers = parsers or {}
        self.sample_every_frames = sample_every_frames
        self.upscale_factor = upscale_factor
        self.min_confidence = min_confidence
        self._states = {name: _ValueState() for name in rois}

    def reset(self) -> None:
        self._states = {name: _ValueState() for name in self.rois}

    def process(self, frame: Any, frame_id: int, timestamp: float) -> list[GameplayEvent]:
        if frame_id % self.sample_every_frames:
            return []
        height, width = frame.shape[:2]
        events: list[GameplayEvent] = []
        for name, roi in self.rois.items():
            x1, y1, x2, y2 = normalized_roi_to_pixels(roi, width, height)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            processed = self.cv2.resize(
                crop,
                None,
                fx=self.upscale_factor,
                fy=self.upscale_factor,
                interpolation=self.cv2.INTER_CUBIC,
            )
            confidence: float | None = None
            if self.backend == "easyocr":
                assert self.reader is not None
                results = self.reader.readtext(
                    processed,
                    detail=1,
                    paragraph=False,
                    allowlist="0123456789",
                    decoder="greedy",
                )
                text = "".join(result[1] for result in results)
                confidence = max((float(result[2]) for result in results), default=None)
            else:
                assert self.pytesseract is not None
                gray = self.cv2.cvtColor(processed, self.cv2.COLOR_BGR2GRAY)
                text = self.pytesseract.image_to_string(
                    gray, config="--psm 7 -c tessedit_char_whitelist=0123456789"
                ).strip()
            parser = self.parsers.get(name, numeric_parser)
            value = parser(text)
            if value is not None and (confidence is None or confidence >= self.min_confidence):
                event = self._update_value(name, value, frame_id, timestamp, confidence)
                if event is not None:
                    events.append(event)
            else:
                self._states[name].candidate = None
                self._states[name].candidate_frames = 0
                self._states[name].confidences = None
        return events

    def _update_value(
        self,
        name: str,
        value: int | float | str,
        frame_id: int,
        timestamp: float,
        confidence: float | None = None,
    ) -> GameplayEvent | None:
        state = self._states[name]
        if state.candidate == value:
            state.candidate_frames += 1
            if confidence is not None:
                assert state.confidences is not None
                state.confidences.append(confidence)
        else:
            state.candidate = value
            state.candidate_frames = 1
            state.confidences = [] if confidence is None else [confidence]
        if state.candidate_frames < self.min_stable_frames or state.confirmed == value:
            return None
        previous = state.confirmed
        state.confirmed = value
        if previous is None:
            return None
        return GameplayEvent(
            timestamp=timestamp,
            frame_id=frame_id,
            event=f"{name}_changed",
            object_id=name,
            object_type=name,
            confidence=min(state.confidences[-self.min_stable_frames:]) if state.confidences else None,
            metadata={
                "previous_value": previous,
                "current_value": value,
                "source": "ocr",
                "confidence_available": confidence is not None,
            },
        )
