import invertransforms as T
from invertransforms.util import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestPad(InvertibleTestCase):

    def test_padding_int(self):
        padding = 100
        tf = T.Pad(padding=padding)
        img_inv = tf(self.img_pil)
        self.assertEqual(img_inv.size, (self.w + 2 * padding, self.h + 2 * padding))
        self.assertEqual(self.img_pil.size, tf.invert(img_inv).size)

    def test_padding_tuple(self):
        pad_lr, pad_tb = 100, 50
        tf = T.Pad(padding=(pad_lr, pad_tb))
        img_inv = tf(self.img_pil)
        self.assertEqual(img_inv.size, (self.w + 2 * pad_lr, self.h + 2 * pad_tb))
        self.assertEqual(self.img_pil.size, tf.invert(img_inv).size)

    def test_invert_before_call(self):
        with self.assertRaises(InvertibleError):
            T.Pad(0).inverse()

    def test_padding_mismatch(self):
        tf = T.Pad(0)
        tf.padding = None
        tf._img_h, tf._img_w = self.img_size
        with self.assertRaises(Exception):
            tf.inverse()
