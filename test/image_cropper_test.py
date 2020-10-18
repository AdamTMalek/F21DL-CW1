import unittest

import src.image_cropper as cropper


class ImageCropperTest(unittest.TestCase):
    def test_crop_image(self):
        test_line = ', '.join([str(i) for i in range(16)])  # 4x4 image
        expected_line = '5, 6, 9, 10'  # 2x2 image
        actual_line = cropper.crop_image(test_line, new_size=2)
        self.assertEquals(expected_line, actual_line)
