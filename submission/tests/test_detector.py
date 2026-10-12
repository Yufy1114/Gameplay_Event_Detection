import unittest
from unittest.mock import patch

import numpy as np

from src.config import ClassFilterConfig
from src.detector import UltralyticsDetector


class _Value:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Coords:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self

    def tolist(self):
        return self.values


class _Box:
    def __init__(self, class_id, confidence, bbox):
        self.cls = _Value(class_id)
        self.conf = _Value(confidence)
        self.xyxy = _Coords(bbox)


class _Model:
    names = {0: "car", 1: "truck"}

    def predict(self, **kwargs):
        result = type("Result", (), {})()
        result.names = self.names
        result.boxes = [
            _Box(0, 0.8, [10, 10, 60, 40]),
            _Box(1, 0.7, [1, 1, 3, 3]),
        ]
        return [result]


class DetectorAdaptationTests(unittest.TestCase):
    def test_mapping_occurs_before_filtering_and_tracking(self) -> None:
        fake_yolo = type("YOLO", (), {"__new__": lambda cls, _: _Model()})
        with patch.dict("sys.modules", {"ultralytics": type("U", (), {"YOLO": fake_yolo})()}):
            detector = UltralyticsDetector(
                "unused.pt",
                target_classes=["car", "truck"],
                class_mapping={"car": "vehicle", "truck": "vehicle"},
                class_filters={"vehicle": ClassFilterConfig(min_bbox_area_ratio=0.01)},
            )
        detections = detector.detect(np.zeros((100, 100, 3)), 0, 0.0)
        self.assertEqual([d.class_name for d in detections], ["vehicle"])

    def test_semantic_nms_suppresses_overlapping_mapped_classes(self) -> None:
        model = _Model()
        model.predict = lambda **kwargs: [type("Result", (), {
            "names": model.names,
            "boxes": [
                _Box(0, 0.8, [10, 10, 60, 40]),
                _Box(1, 0.7, [11, 10, 61, 40]),
            ],
        })()]
        fake_yolo = type("YOLO", (), {"__new__": lambda cls, _: model})
        withilho = type("U", (), {"YOLO": fake_yolo})()
        with patch.dict("sys.modules", {"ultralytics": withilho}):
            detector = UltralyticsDetector(
                "unused.pt",
                target_classes=["car", "truck"],
                class_mapping={"car": "vehicle", "truck": "vehicle"},
                semantic_nms_iou_threshold=0.6,
            )
        detections = detector.detect(np.zeros((100, 100, 3)), 0, 0.0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
