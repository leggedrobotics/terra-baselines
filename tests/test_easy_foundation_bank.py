import unittest

import numpy as np

from scripts.build_easy_foundation_bank import SHAPES, foundation_mask


class EasyFoundationGeometryTests(unittest.TestCase):
    def test_right_angle_rasterization_preserves_analytic_area(self):
        seen = set()
        for shape in SHAPES:
            for seed in range(100):
                mask, geometry = foundation_mask(np.random.default_rng(seed), shape)
                angle = geometry["angle_degrees"]
                if angle % 90:
                    continue
                seen.add((shape, angle))
                width, height = geometry["width_tiles"], geometry["height_tiles"]
                arm = geometry["arm_width_tiles"]
                expected = width * height if arm is None else arm * (width + height - arm)
                with self.subTest(shape=shape, seed=seed, angle=angle):
                    self.assertEqual(int(mask.sum()), expected)
        self.assertEqual(seen, {(shape, angle) for shape in SHAPES for angle in (0, 90, 180, 270)})


if __name__ == "__main__":
    unittest.main()
