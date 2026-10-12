import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.output import write_events_csv, write_events_json
from src.types import GameplayEvent


class _ArrayLike:
    def tolist(self) -> list[int]:
        return [1, 2]


class OutputTests(unittest.TestCase):
    def test_numpy_like_metadata_is_normalized_for_json_and_csv(self) -> None:
        event = GameplayEvent(
            timestamp=1.0,
            frame_id=10,
            event="object_detected",
            object_id=2,
            object_type="object",
            confidence=0.8,
            metadata={"coordinates": _ArrayLike()},
        )
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "events.json"
            csv_path = Path(directory) / "events.csv"
            write_events_json([event], json_path)
            write_events_csv([event], csv_path)
            self.assertEqual(json.loads(json_path.read_text())[0]["metadata"]["coordinates"], [1, 2])
            with csv_path.open(newline="", encoding="utf-8") as handle:
                csv_metadata = json.loads(next(csv.DictReader(handle))["metadata"])
            self.assertEqual(csv_metadata["coordinates"], [1, 2])

    def test_non_finite_event_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "timestamp"):
            GameplayEvent(float("nan"), 1, "event", None, None, 0.5)

    def test_missing_ocr_confidence_serializes_as_json_null_and_blank_csv(self) -> None:
        event = GameplayEvent(1.0, 10, "ammo_changed", "ammo", "ammo", None)
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "events.json"
            csv_path = Path(directory) / "events.csv"
            write_events_json([event], json_path)
            write_events_csv([event], csv_path)
            self.assertIsNone(json.loads(json_path.read_text())[0]["confidence"])
            with csv_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.DictReader(handle))["confidence"], "")


if __name__ == "__main__":
    unittest.main()
