"""Generate the short, video-specific submission demo from validated segments."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path

from .config import PipelineConfig
from .detector import UltralyticsDetector
from .event_engine import EventEngine
from .minimap import HSVMarkerDetector, MinimapAnalyzer
from .tracker import IoUTracker


SECTIONS = (
    ("Character Tracking", 79.0, 86.0, "character"),
    ("Vehicle Tracking", 37.0, 42.5, "vehicle"),
    ("Ammo OCR", 79.0, 86.0, "ammo"),
    ("Minimap Localization", 114.0, 121.5, "minimap"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the 31-second annotated submission demo")
    parser.add_argument("--video", default="gameplay_classic.webm", help="Source gameplay clip")
    parser.add_argument("--config", default="config.example.json", help="Pipeline JSON config")
    parser.add_argument("--events", default="outputs/events.json", help="Existing raw event JSON")
    parser.add_argument("--output", default="outputs/demo.mp4", help="Output MP4 path")
    return parser


def _draw_title(frame, title: str, cv2, bottom: bool = False) -> None:
    top = frame.shape[0] - 82 if bottom else 0
    cv2.rectangle(frame, (0, top), (frame.shape[1], top + 82), (0, 0, 0), -1)
    cv2.putText(frame, title, (35, top + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2)


def _draw_recent(frame, recent, cv2, start_y: int = 118) -> None:
    for row, label in enumerate(recent):
        cv2.putText(frame, label, (35, start_y + row * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def _write_title_card(writer, title: str, size: tuple[int, int], fps: float, cv2) -> None:
    import numpy as np

    card = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)[0]
    origin = ((size[0] - text_size[0]) // 2, (size[1] + text_size[1]) // 2)
    cv2.putText(card, title, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    for _ in range(round(fps)):
        writer.write(card)


def build_demo(video_path: str, config_path: str, events_path: str, output_path: str) -> None:
    import cv2

    config = PipelineConfig.load(config_path)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise OSError(f"Video cannot be opened: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Invalid source video metadata")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Video writer cannot be opened: {output}")

    raw_events = json.loads(Path(events_path).read_text(encoding="utf-8"))
    detector = UltralyticsDetector(
        model_name=config.detector.model_name,
        confidence_threshold=config.detector.confidence_threshold,
        image_size=config.detector.image_size,
        target_classes=config.detector.target_classes,
        class_mapping=config.detector.class_mapping,
        class_filters=config.detector.class_filters,
        semantic_nms_iou_threshold=config.detector.semantic_nms_iou_threshold,
        device=config.detector.device,
    )
    minimap = MinimapAnalyzer(
        config.minimap.roi,
        HSVMarkerDetector(
            config.minimap.hsv_lower,
            config.minimap.hsv_upper,
            config.minimap.marker_min_area,
            config.minimap.marker_max_area,
            config.minimap.marker_min_circularity,
            config.minimap.border_min_pixels,
        ),
    ) if config.minimap.roi else None

    try:
        for title, start, end, mode in SECTIONS:
            _write_title_card(writer, title, (width, height), fps, cv2)
            capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            tracker = IoUTracker(config.tracker.iou_matching_threshold, config.tracker.max_missing_frames)
            engine = EventEngine(
                config.temporal_smoothing.min_confirmation_frames,
                config.temporal_smoothing.disappearance_frames,
            )
            recent: deque[str] = deque(maxlen=4)
            ammo_events = [event for event in raw_events if event["event"] == "ammo_changed"]
            ammo_value = ammo_events[0].get("metadata", {}).get("previous_value") if ammo_events else None
            consumed_ammo: set[int] = set()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_id = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                timestamp = frame_id / fps
                if timestamp >= end:
                    break
                if mode in {"character", "vehicle"}:
                    detections = [
                        detection for detection in detector.detect(frame, frame_id, timestamp)
                        if detection.class_name == mode
                    ]
                    tracked = tracker.update(detections, frame_id)
                    frame_events = engine.update(tracked, frame_id, timestamp)
                    tracker.retire({e.object_id for e in frame_events if e.event.endswith("_disappeared")})
                    for event in frame_events:
                        recent.appendleft(f"{event.timestamp:.1f}s {event.event}")
                    for obj in tracked:
                        x1, y1, x2, y2 = (round(value) for value in obj.bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 220, 50), 2)
                        cv2.putText(frame, f"{obj.class_name} | ID {obj.track_id} | {obj.confidence:.2f}",
                                    (x1, max(100, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 50), 2)
                    _draw_recent(frame, recent, cv2)
                elif mode == "ammo":
                    for index, event in enumerate(ammo_events):
                        if index not in consumed_ammo and event["timestamp"] <= timestamp:
                            ammo_value = event.get("metadata", {}).get("current_value")
                            recent.appendleft(f"{event['timestamp']:.1f}s ammo_changed")
                            consumed_ammo.add(index)
                    if ammo_value is not None:
                        cv2.putText(frame, f"Confirmed ammo: {ammo_value}", (35, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    _draw_recent(frame, recent, cv2, start_y=180)
                    x1, y1, x2, y2 = (round(v) for v in (688, 1025, 715, 1068))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                elif mode == "minimap" and minimap is not None:
                    x1, y1, x2, y2 = (round(v) for v in (1365, 0, 1655, 300))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    objects = minimap.analyze(frame)
                    if objects:
                        marker = objects[0]
                        px = x1 + round(marker.minimap_x * (x2 - x1))
                        py = y1 + round(marker.minimap_y * (y2 - y1))
                        cv2.circle(frame, (px, py), 10, (0, 0, 255), 3)
                        cv2.putText(frame, f"Map marker: ({marker.minimap_x:.2f}, {marker.minimap_y:.2f})",
                                    (35, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 200, 0), 2)
                _draw_title(frame, title, cv2, bottom=mode == "minimap")
                writer.write(frame)
    finally:
        capture.release()
        writer.release()


def main() -> int:
    args = _parser().parse_args()
    build_demo(args.video, args.config, args.events, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
