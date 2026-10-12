"""Command-line pipeline orchestration."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Sequence

from .config import ConfigurationError, PipelineConfig
from .detector import DetectorError, UltralyticsDetector
from .event_engine import EventEngine
from .minimap import HSVMarkerDetector, MinimapAnalyzer
from .ocr import OCRProcessor
from .output import write_events_csv, write_events_json
from .tracker import IoUTracker
from .types import GameplayEvent
from .visualizer import VideoVisualizer

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect structured gameplay events from video")
    parser.add_argument("--video", required=True, help="Input gameplay video path")
    parser.add_argument("--config", default="config.example.json", help="Pipeline JSON config")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def run_pipeline(video_path: str | Path, config: PipelineConfig) -> list[GameplayEvent]:
    config.validate()
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to process video") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Video cannot be opened: {video_path}")
    visualizer: VideoVisualizer | None = None
    events: list[GameplayEvent] = []
    frame_id = 0
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS is unavailable or invalid")
        width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (
            not math.isfinite(width_value)
            or not math.isfinite(height_value)
            or width_value <= 0
            or height_value <= 0
        ):
            raise ValueError("Video frame dimensions are unavailable")
        width, height = int(width_value), int(height_value)

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
        tracker: IoUTracker | None = None
        event_engine: EventEngine | None = None
        ocr = (
            OCRProcessor(
                config.ocr.rois,
                config.ocr.min_stable_frames,
                backend=config.ocr.backend,
                sample_every_frames=config.ocr.sample_every_frames,
                upscale_factor=config.ocr.upscale_factor,
                min_confidence=config.ocr.min_confidence,
            )
            if config.ocr.enabled
            else None
        )
        minimap = None
        if config.minimap.enabled:
            assert config.minimap.roi is not None  # Guaranteed by config.validate().
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
            )
        visualizer = (
            VideoVisualizer(
                config.output.video_path,
                fps,
                (width, height),
                config.output.show_recent_events,
            )
            if config.output.visualization_enabled and config.output.video_path
            else None
        )
        segments = config.analysis_segments
        segment_index = 0
        active_segment = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_id / fps
            if segments:
                while segment_index < len(segments) and timestamp >= segments[segment_index].end:
                    segment_index += 1
                    tracker = None
                    event_engine = None
                if segment_index >= len(segments):
                    break
                candidate = segments[segment_index]
                if timestamp < candidate.start:
                    frame_id += 1
                    continue
                if active_segment is not candidate:
                    active_segment = candidate
                    tracker = None
                    event_engine = None
                    if ocr is not None:
                        ocr.reset()
                    LOGGER.info(
                        "Analyzing segment %.3f-%.3fs (%s)",
                        candidate.start,
                        candidate.end,
                        candidate.purpose,
                    )
            if tracker is None:
                tracker = IoUTracker(
                    iou_threshold=config.tracker.iou_matching_threshold,
                    max_missing_frames=config.tracker.max_missing_frames,
                )
                event_engine = EventEngine(
                    min_confirmation_frames=config.temporal_smoothing.min_confirmation_frames,
                    disappearance_frames=config.temporal_smoothing.disappearance_frames,
                )
            detections = detector.detect(frame, frame_id, timestamp)
            tracked = tracker.update(detections, frame_id)
            assert event_engine is not None
            frame_events = event_engine.update(tracked, frame_id, timestamp)
            disappeared_ids = {
                event.object_id
                for event in frame_events
                if event.event.endswith("_disappeared") and isinstance(event.object_id, int)
            }
            tracker.retire(disappeared_ids)
            if ocr is not None:
                frame_events.extend(ocr.process(frame, frame_id, timestamp))
            if minimap is not None:
                # Marker interpretation is intentionally adapter-specific; no game semantics are assumed.
                minimap.analyze(frame)
            events.extend(frame_events)
            if visualizer is not None:
                visualizer.write(frame, tracked, frame_events)
            frame_id += 1
    finally:
        capture.release()
        if visualizer is not None:
            visualizer.close()

    if frame_id == 0:
        LOGGER.warning("Input video contained no readable frames")
    if config.output.json_path:
        write_events_json(events, config.output.json_path)
    if config.output.csv_path:
        write_events_csv(events, config.output.csv_path)
    LOGGER.info("Processed %d frames and wrote %d events", frame_id, len(events))
    return events


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        config = PipelineConfig.load(args.config)
        run_pipeline(args.video, config)
    except (ConfigurationError, DetectorError, OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
