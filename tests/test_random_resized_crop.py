import invertransforms as T
from invertransforms.lib import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestRandomResizedCrop(InvertibleTestCase):

    def test_deprecated(self):
        with self.assertWarns(UserWarning):
            tf = T.RandomSizedCrop(self.crop_size)
            self.assertIsInstance(tf, T.RandomResizedCrop)

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleError):
            T.RandomResizedCrop(self.crop_size).inverse()

    def test_invert(self):
        tf = T.RandomResizedCrop(self.crop_size)
        img_inv = tf(self.img_pil)
        self.assertEqual(img_inv.size[::-1], self.crop_size)

        img_pil = tf.invert(img_inv)
        self.assertEqual(img_pil.size, self.img_pil.size)
