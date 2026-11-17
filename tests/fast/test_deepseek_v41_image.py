# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import math
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase
from modelbuilder.helpers.deepseek_v41_image import DeepseekV41ImageProcessor


class TestDeepseekV41ImageProcessor(ExtTestCase):
    def test_patch_order_and_row_delimiters(self):
        processor = DeepseekV41ImageProcessor(dict(patch_size=2, downsample_ratio=2, max_image_tokens=32, min_pixels=0))
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        image[:2, :2] = 255
        image[2:, 4:] = 127
        patches, grid, types = processor(image)
        np.testing.assert_array_equal(grid, [2, 3])
        self.assertEqual(patches.shape, (6, 3, 2, 2))
        np.testing.assert_array_equal(patches[0], np.ones((3, 2, 2)))
        np.testing.assert_array_equal(patches[1:5], -np.ones((4, 3, 2, 2)))
        np.testing.assert_allclose(patches[5], 127 / 127.5 - 1, atol=1e-7)
        np.testing.assert_array_equal(types, [0, 1, 1, 2, 3])

    def test_image_token_budget_including_extreme_aspect_ratios(self):
        processor = DeepseekV41ImageProcessor(dict(patch_size=14, downsample_ratio=3, max_image_tokens=64, min_pixels=295936))
        for width, height in ((1, 1), (12, 20000), (20000, 12), (4000, 3000)):
            with self.subTest(width=width, height=height):
                rows, columns = processor.plan(width, height)
                self.assertGreater(rows, 0)
                self.assertGreater(columns, 0)
                count = math.ceil(rows / 3) * (math.ceil(columns / 3) + 1) + 2
                self.assertLessEqual(count, 64)
        with self.assertRaisesRegex(ValueError, "positive"):
            processor.plan(0, 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
