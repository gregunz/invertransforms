import invertransforms.color_transforms
import invertransforms.random_erasing
from tests.invertible_test_case import InvertibleTestCase


class TestRandomErasing(InvertibleTestCase):

    def test_invert_is_identity(self):
        tf = invertransforms.random_erasing.RandomErasing()
        tf_inv = tf.inverse()
        self.assertEqual(self.n, tf_inv(self.n))
