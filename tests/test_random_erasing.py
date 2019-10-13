import invertransforms as T
from tests.invertible_test_case import InvertibleTestCase


class TestRandomErasing(InvertibleTestCase):

    def test_invert_is_identity(self):
        tf = T.RandomErasing()
        tf_inv = tf.invert()
        self.assertEqual(self.n, tf_inv(self.n))
