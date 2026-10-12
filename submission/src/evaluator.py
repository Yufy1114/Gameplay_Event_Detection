"""Detection- and event-level precision/recall evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tracker import calculate_iou
from .types import Detection, GameplayEvent


@dataclass(frozen=True, slots=True)
class Metrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int) -> "Metrics":
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return cls(tp, fp, fn, precision, recall, f1)


def partition_segment_initialization_events(
    predictions: list[GameplayEvent],
    segment_starts: list[float],
    initialization_window_seconds: float,
) -> tuple[list[GameplayEvent], list[dict[str, Any]]]:
    """Separate deterministic segment-initialization events from scored predictions.

    The rule is deliberately independent of matching: every prediction whose
    timestamp falls in ``[segment_start, segment_start + window]`` is excluded,
    regardless of event type or whether it would otherwise match ground truth.
    Raw pipeline events are not modified.
    """
    if (
        not isinstance(initialization_window_seconds, (int, float))
        or isinstance(initialization_window_seconds, bool)
        or not math.isfinite(initialization_window_seconds)
        or initialization_window_seconds < 0
    ):
        raise ValueError("initialization_window_seconds must be finite and non-negative")
    if any(
        not isinstance(start, (int, float))
        or isinstance(start, bool)
        or not math.isfinite(start)
        or start < 0
        for start in segment_starts
    ):
        raise ValueError("segment_starts must contain finite non-negative numbers")

    scored: list[GameplayEvent] = []
    excluded: list[dict[str, Any]] = []
    for prediction in predictions:
        boundary = next(
            (
                float(start)
                for start in segment_starts
                if float(start)
                <= prediction.timestamp
                <= float(start) + initialization_window_seconds
            ),
            None,
        )
        if boundary is None:
            scored.append(prediction)
            continue
        record = prediction.to_dict()
        record.update(
            {
                "reason": "segment_initialization",
                "segment_start": boundary,
                "initialization_window_seconds": initialization_window_seconds,
            }
        )
        excluded.append(record)
    return scored, excluded


def evaluate_detections(
    predictions: list[Detection],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Metrics:
    """Match predictions and annotations one-to-one by frame, class, and IoU."""
    if (
        not isinstance(iou_threshold, (int, float))
        or isinstance(iou_threshold, bool)
        or not math.isfinite(iou_threshold)
        or not 0 <= iou_threshold <= 1
    ):
        raise ValueError("iou_threshold must be a finite number in [0, 1]")
    gt_by_frame: dict[int, list[dict[str, Any]]] = {}
    for frame in ground_truth:
        gt_by_frame.setdefault(int(frame["frame_id"]), []).extend(frame.get("objects", []))
    pred_by_frame: dict[int, list[Detection]] = {}
    for prediction in predictions:
        pred_by_frame.setdefault(prediction.frame_id, []).append(prediction)

    tp = fp = fn = 0
    for frame_id in set(gt_by_frame) | set(pred_by_frame):
        frame_predictions = pred_by_frame.get(frame_id, [])
        frame_ground_truth = gt_by_frame.get(frame_id, [])
        candidates: list[tuple[float, int, int]] = []
        for pred_index, prediction in enumerate(frame_predictions):
            for gt_index, annotation in enumerate(frame_ground_truth):
                if prediction.class_name != annotation.get("class_name"):
                    continue
                score = calculate_iou(prediction.bbox, tuple(annotation["bbox"]))
                if score >= iou_threshold:
                    candidates.append((score, pred_index, gt_index))
        matched_predictions: set[int] = set()
        matched_ground_truth: set[int] = set()
        for _, pred_index, gt_index in sorted(candidates, reverse=True):
            if pred_index in matched_predictions or gt_index in matched_ground_truth:
                continue
            matched_predictions.add(pred_index)
            matched_ground_truth.add(gt_index)
        tp += len(matched_predictions)
        fp += len(frame_predictions) - len(matched_predictions)
        fn += len(frame_ground_truth) - len(matched_ground_truth)
    return Metrics.from_counts(tp, fp, fn)


def evaluate_events(
    predictions: list[GameplayEvent],
    ground_truth: list[dict[str, Any]],
    tolerance_seconds: float = 0.5,
) -> Metrics:
    """Match events one-to-one by event type and closest timestamp in a tolerance."""
    if (
        not isinstance(tolerance_seconds, (int, float))
        or isinstance(tolerance_seconds, bool)
        or not math.isfinite(tolerance_seconds)
        or tolerance_seconds < 0
    ):
        raise ValueError("tolerance_seconds must be a finite non-negative number")
    candidates: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, annotation in enumerate(ground_truth):
            if prediction.event != annotation.get("event"):
                continue
            difference = abs(prediction.timestamp - float(annotation["timestamp"]))
            if difference <= tolerance_seconds:
                candidates.append((difference, pred_index, gt_index))
    matched_predictions: set[int] = set()
    matched_ground_truth: set[int] = set()
    for _, pred_index, gt_index in sorted(candidates):
        if pred_index in matched_predictions or gt_index in matched_ground_truth:
            continue
        matched_predictions.add(pred_index)
        matched_ground_truth.add(gt_index)
    return Metrics.from_counts(
        len(matched_predictions),
        len(predictions) - len(matched_predictions),
        len(ground_truth) - len(matched_ground_truth),
    )


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load evaluation data from {path}: {exc}") from exc
    if not isinstance(value, list):
        raise ValueError(f"Evaluation data must be a JSON list: {path}")
    return value
