import unittest

from src.evaluator import (
    Metrics,
    evaluate_detections,
    evaluate_events,
    partition_segment_initialization_events,
)
from src.types import Detection, GameplayEvent


class EvaluatorTests(unittest.TestCase):
    def test_precision_recall_f1(self) -> None:
        metrics = Metrics.from_counts(tp=2, fp=1, fn=2)
        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertAlmostEqual(metrics.recall, 1 / 2)
        self.assertAlmostEqual(metrics.f1, 4 / 7)

    def test_detection_matching_is_class_aware(self) -> None:
        predictions = [
            Detection(10, 1.0, "class_a", 0.9, (0, 0, 10, 10)),
            Detection(10, 1.0, "wrong_class", 0.8, (20, 20, 30, 30)),
        ]
        ground_truth = [{
            "frame_id": 10,
            "objects": [
                {"class_name": "class_a", "bbox": [0, 0, 10, 10]},
                {"class_name": "class_b", "bbox": [20, 20, 30, 30]},
            ],
        }]
        metrics = evaluate_detections(predictions, ground_truth, iou_threshold=0.5)
        self.assertEqual((metrics.true_positives, metrics.false_positives, metrics.false_negatives), (1, 1, 1))

    def test_event_matching_uses_type_and_time_tolerance(self) -> None:
        predictions = [GameplayEvent(1.2, 12, "class_a_detected", 1, "class_a", 0.9)]
        ground_truth = [{"timestamp": 1.0, "event": "class_a_detected"}]
        metrics = evaluate_events(predictions, ground_truth, tolerance_seconds=0.25)
        self.assertEqual((metrics.true_positives, metrics.false_positives, metrics.false_negatives), (1, 0, 0))

    def test_duplicate_frame_entries_do_not_overwrite_ground_truth(self) -> None:
        predictions = [Detection(4, 0.4, "class_a", 0.9, (0, 0, 10, 10))]
        ground_truth = [
            {"frame_id": 4, "objects": [{"class_name": "class_a", "bbox": [0, 0, 10, 10]}]},
            {"frame_id": 4, "objects": [{"class_name": "class_b", "bbox": [20, 20, 30, 30]}]},
        ]
        metrics = evaluate_detections(predictions, ground_truth)
        self.assertEqual((metrics.true_positives, metrics.false_positives, metrics.false_negatives), (1, 0, 1))

    def test_detection_matching_is_one_to_one(self) -> None:
        predictions = [
            Detection(1, 0.1, "class_a", 0.9, (0, 0, 10, 10)),
            Detection(1, 0.1, "class_a", 0.8, (0, 0, 10, 10)),
        ]
        ground_truth = [
            {"frame_id": 1, "objects": [{"class_name": "class_a", "bbox": [0, 0, 10, 10]}]}
        ]
        metrics = evaluate_detections(predictions, ground_truth)
        self.assertEqual((metrics.true_positives, metrics.false_positives), (1, 1))

    def test_event_matching_is_one_to_one(self) -> None:
        predictions = [
            GameplayEvent(1.0, 10, "class_a_detected", 1, "class_a", 0.9),
            GameplayEvent(1.1, 11, "class_a_detected", 2, "class_a", 0.8),
        ]
        ground_truth = [{"timestamp": 1.0, "event": "class_a_detected"}]
        metrics = evaluate_events(predictions, ground_truth, tolerance_seconds=0.5)
        self.assertEqual((metrics.true_positives, metrics.false_positives), (1, 1))

    def test_initialization_exclusion_is_deterministic_and_type_agnostic(self) -> None:
        predictions = [
            GameplayEvent(10.1, 101, "class_a_detected", 1, "class_a", 0.9),
            GameplayEvent(10.2, 102, "ammo_changed", "ammo", "ammo", 0.8),
            GameplayEvent(10.3, 103, "class_a_detected", 2, "class_a", 0.7),
        ]
        scored, excluded = partition_segment_initialization_events(
            predictions, [10.0], initialization_window_seconds=0.2
        )
        self.assertEqual([event.frame_id for event in scored], [103])
        self.assertEqual([event["frame_id"] for event in excluded], [101, 102])
        self.assertTrue(all(event["reason"] == "segment_initialization" for event in excluded))

    def test_initialization_exclusion_does_not_depend_on_ground_truth_matching(self) -> None:
        predictions = [
            GameplayEvent(5.1, 51, "class_a_detected", 1, "class_a", 0.9),
            GameplayEvent(5.5, 55, "class_a_detected", 2, "class_a", 0.8),
        ]
        scored, _ = partition_segment_initialization_events(predictions, [5.0], 0.2)
        metrics = evaluate_events(
            scored,
            [{"timestamp": 5.1, "event": "class_a_detected"}],
            tolerance_seconds=0.2,
        )
        self.assertEqual((metrics.true_positives, metrics.false_positives, metrics.false_negatives), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
