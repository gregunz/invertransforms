import math

import invertransforms as T
from invertransforms.util import InvertibleException
from tests.invertible_test_case import InvertibleTestCase


class TestRandomRotCrop(InvertibleTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.crop_size = (min(self.crop_size), min(self.crop_size))
        self.tf = T.RandomRotCrop(crop_size=self.crop_size)

    def test_init(self):
        with self.assertRaises(ValueError):
            T.RandomRotCrop(-1)

    def test_call(self):
        tf = T.RandomRotCrop()
        self.assertGreaterEqual(min(tf(self.img_pil).size), int(min(self.img_pil.size) / math.sqrt(2)))
        self.assertGreaterEqual(max(self.img_pil.size), max(tf(self.img_pil).size))

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleException):
            T.RandomRotCrop(crop_size=self.crop_size).invert()

    def test_invert(self):
        tf = T.RandomRotCrop(crop_size=self.crop_size)
        img_inv = tf(self.img_pil)
        self.assertEqual(img_inv.size[::-1], self.crop_size)

        img_pil = tf.inverse(img_inv)
        self.assertEqual(self.img_pil.size, img_pil.size)

    def test_bigger_cropping(self):
        tf = T.RandomRotCrop(degrees=[5, 40], crop_size=max(self.img_size))
        with self.assertWarns(UserWarning):
            tf(self.img_pil)
