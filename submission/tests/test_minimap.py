import unittest

import cv2
import numpy as np

from src.minimap import HSVMarkerDetector, MinimapAnalyzer


class MinimapTests(unittest.TestCase):
    def test_marker_is_normalized_and_absence_returns_no_observation(self) -> None:
        detector = HSVMarkerDetector((22, 120, 120), (38, 255, 255), 200, 1000, 0.6, 20)
        analyzer = MinimapAnalyzer((0.0, 0.0, 1.0, 1.0), detector)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(image, (40, 50), 12, (0, 255, 255), -1)
        cv2.line(image, (95, 0), (95, 99), (0, 255, 255), 3)
        found = analyzer.analyze(image)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0].minimap_x, 0.4, places=1)
        self.assertEqual(analyzer.analyze(np.zeros_like(image)), [])


if __name__ == "__main__":
    unittest.main()
