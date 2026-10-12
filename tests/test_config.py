import json
import tempfile
import unittest
from pathlib import Path

from src.config import ConfigurationError, PipelineConfig


class ConfigTests(unittest.TestCase):
    def _load(self, value: dict) -> PipelineConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return PipelineConfig.load(path)

    def test_tracker_must_outlive_disappearance_window(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "max_missing_frames"):
            self._load({
                "tracker": {"max_missing_frames": 2},
                "temporal_smoothing": {"disappearance_frames": 3},
            })

    def test_wrong_threshold_type_has_configuration_error(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "confidence_threshold"):
            self._load({"detector": {"confidence_threshold": "high"}})

    def test_rois_must_be_an_object(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "ocr.rois"):
            self._load({"ocr": {"rois": []}})

    def test_target_classes_must_be_a_list(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "target_classes"):
            self._load({"detector": {"target_classes": "person"}})

    def test_non_finite_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "Non-finite"):
            self._load({"detector": {"confidence_threshold": float("nan")}})

    def test_semantic_mapping_and_segments_load(self) -> None:
        config = self._load({
            "detector": {
                "target_classes": ["person", "car"],
                "class_mapping": {"person": "character", "car": "vehicle"},
                "class_filters": {"vehicle": {"min_bbox_area_ratio": 0.002}},
            },
            "analysis_segments": [{"start": 1.0, "end": 2.0, "purpose": "test"}],
        })
        self.assertEqual(config.detector.class_mapping["person"], "character")
        self.assertEqual(config.analysis_segments[0].purpose, "test")

    def test_overlapping_segments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "non-overlapping"):
            self._load({"analysis_segments": [
                {"start": 1.0, "end": 3.0, "purpose": "first"},
                {"start": 2.0, "end": 4.0, "purpose": "second"},
            ]})


if __name__ == "__main__":
    unittest.main()
