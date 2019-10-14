import invertransforms as T
from tests.invertible_test_case import InvertibleTestCase


class TestColorJitter(InvertibleTestCase):
    def test_invert_is_identity(self):
        tf = T.ColorJitter()
        img_tf = tf(self.img_pil)
        self.assertEqual(img_tf.size, self.img_pil.size)
        tf_inv = tf.inverse()
        self.assertEqual(self.n, tf_inv(self.n))
