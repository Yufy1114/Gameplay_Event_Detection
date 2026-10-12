import unittest

from src.ocr import OCRProcessor, _ValueState


class OCRTemporalTests(unittest.TestCase):
    def test_three_consistent_values_are_required_and_confidence_is_real(self) -> None:
        processor = object.__new__(OCRProcessor)
        processor.min_stable_frames = 3
        processor._states = {"ammo": _ValueState()}
        self.assertIsNone(processor._update_value("ammo", 30, 1, 0.1, 0.9))
        self.assertIsNone(processor._update_value("ammo", 30, 2, 0.2, 0.8))
        self.assertIsNone(processor._update_value("ammo", 30, 3, 0.3, 0.7))
        processor._update_value("ammo", 29, 4, 0.4, 0.9)
        processor._update_value("ammo", 29, 5, 0.5, 0.8)
        event = processor._update_value("ammo", 29, 6, 0.6, 0.7)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.event, "ammo_changed")
        self.assertEqual(event.metadata["current_value"], 29)
        self.assertEqual(event.confidence, 0.7)


if __name__ == "__main__":
    unittest.main()
