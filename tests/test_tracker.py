import unittest

from src.tracker import IoUTracker, calculate_iou
from src.event_engine import EventEngine
from src.types import Detection


class IoUTests(unittest.TestCase):
    def test_iou_overlap_and_disjoint_boxes(self) -> None:
        self.assertAlmostEqual(calculate_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(calculate_iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)

    def test_tracker_preserves_id_across_short_gap(self) -> None:
        tracker = IoUTracker(iou_threshold=0.3, max_missing_frames=3)
        first = Detection(0, 0.0, "generic_object", 0.9, (0, 0, 10, 10))
        later = Detection(2, 0.2, "generic_object", 0.8, (1, 0, 11, 10))
        first_id = tracker.update([first], 0)[0].track_id
        tracker.update([], 1)
        self.assertEqual(tracker.update([later], 2)[0].track_id, first_id)

    def test_tracker_retention_and_event_disappearance_boundary_are_consistent(self) -> None:
        tracker = IoUTracker(iou_threshold=0.3, max_missing_frames=2)
        engine = EventEngine(min_confirmation_frames=1, disappearance_frames=2)
        detection = Detection(0, 0.0, "generic_object", 0.9, (0, 0, 10, 10))
        tracks = tracker.update([detection], 0)
        engine.update(tracks, 0, 0.0)
        self.assertEqual(engine.update(tracker.update([], 1), 1, 0.1), [])
        events = engine.update(tracker.update([], 2), 2, 0.2)
        self.assertEqual([event.event for event in events], ["generic_object_disappeared"])

    def test_disappeared_lifecycle_can_retire_long_lived_tracker_id(self) -> None:
        tracker = IoUTracker(iou_threshold=0.3, max_missing_frames=10)
        first = Detection(0, 0.0, "generic_object", 0.9, (0, 0, 10, 10))
        old_id = tracker.update([first], 0)[0].track_id
        tracker.retire({old_id})
        reappeared = Detection(1, 0.1, "generic_object", 0.9, (0, 0, 10, 10))
        self.assertNotEqual(tracker.update([reappeared], 1)[0].track_id, old_id)


if __name__ == "__main__":
    unittest.main()
