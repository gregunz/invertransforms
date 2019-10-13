import invertransforms as T
from invertransforms.util import InvertibleException
from tests.invertible_test_case import InvertibleTestCase


class TestGrayscale(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleException):
            T.RandomGrayscale().invert()

    def test_invert_is_identity(self):
        tf = T.Grayscale()
        tf_inv = tf.invert()
        self.assertEqual(self.n, tf_inv(self.n))

        tf_random_0 = T.RandomGrayscale(p=0)
        self.assertEqual(self.n, tf_random_0(self.n))
        self.assertEqual(self.n, tf_random_0.inverse(self.n))

        tf_random_1 = T.RandomGrayscale(p=1)
        tf_random_1(self.img_pil)
        self.assertEqual(self.n, tf_random_1.inverse(self.n))
