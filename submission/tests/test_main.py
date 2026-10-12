import sys
import types
import unittest
from unittest.mock import patch

from src.config import PipelineConfig
from src.main import run_pipeline


class _Capture:
    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return self.fps if property_id == 1 else 100.0

    def release(self) -> None:
        self.released = True


class MainPipelineTests(unittest.TestCase):
    def test_nan_fps_is_rejected_and_capture_is_released(self) -> None:
        capture = _Capture(float("nan"))
        fake_cv2 = types.SimpleNamespace(
            CAP_PROP_FPS=1,
            CAP_PROP_FRAME_WIDTH=2,
            CAP_PROP_FRAME_HEIGHT=3,
            VideoCapture=lambda _: capture,
        )
        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            with self.assertRaisesRegex(ValueError, "FPS"):
                run_pipeline("unused.mp4", PipelineConfig())
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
