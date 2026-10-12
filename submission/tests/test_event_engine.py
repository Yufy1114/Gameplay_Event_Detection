import unittest

from src.event_engine import EventEngine
from src.types import TrackedObject


def tracked(frame_id: int, track_id: int = 7) -> TrackedObject:
    return TrackedObject(frame_id, frame_id / 10, "generic_object", 0.9, (1, 2, 10, 20), track_id)


class EventEngineTests(unittest.TestCase):
    def test_temporal_confirmation_and_deduplication(self) -> None:
        engine = EventEngine(min_confirmation_frames=3, disappearance_frames=2)
        self.assertEqual(engine.update([tracked(0)], 0, 0.0), [])
        self.assertEqual(engine.update([tracked(1)], 1, 0.1), [])
        events = engine.update([tracked(2)], 2, 0.2)
        self.assertEqual([event.event for event in events], ["generic_object_detected"])
        self.assertEqual(engine.update([tracked(3)], 3, 0.3), [])
        self.assertEqual(engine.update([tracked(4)], 4, 0.4), [])

    def test_single_miss_does_not_disappear(self) -> None:
        engine = EventEngine(min_confirmation_frames=1, disappearance_frames=2)
        engine.update([tracked(0)], 0, 0.0)
        self.assertEqual(engine.update([], 1, 0.1), [])
        self.assertEqual(engine.update([tracked(2)], 2, 0.2), [])

    def test_disappearance_after_configured_gap_is_emitted_once(self) -> None:
        engine = EventEngine(min_confirmation_frames=1, disappearance_frames=2)
        engine.update([tracked(0)], 0, 0.0)
        self.assertEqual(engine.update([], 1, 0.1), [])
        events = engine.update([], 2, 0.2)
        self.assertEqual([event.event for event in events], ["generic_object_disappeared"])
        self.assertEqual(engine.update([], 3, 0.3), [])

    def test_unconfirmed_track_never_generates_disappearance(self) -> None:
        engine = EventEngine(min_confirmation_frames=3, disappearance_frames=2)
        engine.update([tracked(0)], 0, 0.0)
        engine.update([tracked(1)], 1, 0.1)
        self.assertEqual(engine.update([], 3, 0.3), [])


if __name__ == "__main__":
    unittest.main()
