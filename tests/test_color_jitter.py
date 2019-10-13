import invertransforms as T
from tests.invertible_test_case import InvertibleTestCase


class TestColorJitter(InvertibleTestCase):
    def test_invert_is_identity(self):
        tf = T.ColorJitter()
        tf_inv = tf.invert()
        self.assertEqual(self.n, tf_inv(self.n))
