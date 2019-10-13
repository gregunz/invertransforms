import invertransforms as T
from invertransforms.util import InvertibleException
from tests.invertible_test_case import InvertibleTestCase


class TestRandomOrder(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleException):
            T.RandomOrder([]).invert()

    def test_invert(self):
        tf = T.RandomOrder([
            T.CenterCrop(self.crop_size),
            T.RandomHorizontalFlip(1),
            T.RandomVerticalFlip(1),
        ])
        img_inv = tf(self.img_pil)
        img_pil = tf.inverse(img_inv)
        self.assertEqual(img_pil.size, self.img_pil.size)
        center = (self.w // 2 - 10, self.h // 2 + 10)
        self.assertEqual(img_pil.getpixel(center), self.img_pil.getpixel(center))
