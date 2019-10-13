import invertransforms as T
from invertransforms.util import InvertibleException
from tests.invertible_test_case import InvertibleTestCase


class TestRandomResizedCrop(InvertibleTestCase):

    def test_deprecated(self):
        with self.assertWarns(UserWarning):
            tf = T.RandomSizedCrop(self.crop_size)
            self.assertIsInstance(tf, T.RandomResizedCrop)

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleException):
            T.RandomResizedCrop(self.crop_size).invert()

    def test_invert(self):
        tf = T.RandomResizedCrop(self.crop_size)
        img_inv = tf(self.img_pil)
        self.assertEqual(img_inv.size[::-1], self.crop_size)

        img_pil = tf.inverse(img_inv)
        self.assertEqual(img_pil.size, self.img_pil.size)
