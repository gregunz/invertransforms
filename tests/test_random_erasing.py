import invertransforms as T
from tests.invertible_test_case import InvertibleTestCase


class TestRandomErasing(InvertibleTestCase):

    def test_invert_is_identity(self):
        tf = T.RandomErasing()
        img_tf = tf(self.img_tensor)
        self.assertEqual(img_tf.size(), self.img_tensor.size())
        tf_inv = tf.inverse()
        self.assertEqual(self.n, tf_inv(self.n))
