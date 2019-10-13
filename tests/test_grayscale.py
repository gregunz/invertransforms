import invertransforms.color_transforms
from invertransforms.util import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestGrayscale(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleError):
            invertransforms.color_transforms.RandomGrayscale().inverse()

    def test_invert_is_identity(self):
        tf = invertransforms.color_transforms.Grayscale()
        tf_inv = tf.inverse()
        self.assertEqual(self.n, tf_inv(self.n))

        tf_random_0 = invertransforms.color_transforms.RandomGrayscale(p=0)
        self.assertEqual(self.n, tf_random_0(self.n))
        self.assertEqual(self.n, tf_random_0.invert(self.n))

        tf_random_1 = invertransforms.color_transforms.RandomGrayscale(p=1)
        tf_random_1(self.img_pil)
        self.assertEqual(self.n, tf_random_1.invert(self.n))
