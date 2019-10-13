import invertransforms as T
from invertransforms.util import InvertibleException
from tests.invertible_test_case import InvertibleTestCase


class TestRotation(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleException):
            T.RandomRotation(degrees=180).invert()

        with self.assertRaises(InvertibleException):
            T.Rotation(angle=180, expand=True).invert()

    def test_repr(self):
        angle = 20
        center = (0, 0)
        tf = T.Rotation(angle=angle, center=center)
        self.assertIn('Rotation', repr(tf))
        self.assertIn(f'angle={angle}', repr(tf))
        self.assertIn(f'center={center}', repr(tf))

    def test_invert(self):
        tf = T.RandomRotation(degrees=180, expand=True)
        img_inv = tf(self.img_pil)
        tf_inv = tf.invert()
        self.assertIsInstance(tf_inv, T.Rotation)

        img = tf_inv(img_inv)
        self.assertEqual(img.size, self.img_pil.size)
