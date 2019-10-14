import invertransforms as T
from invertransforms.lib import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestRotation(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleError):
            T.RandomRotation(degrees=180).inverse()

        with self.assertRaises(InvertibleError):
            T.Rotation(angle=180, expand=True).inverse()

    def test_call(self):
        tf = T.Rotation(angle=180, expand=True)
        img_tf = tf(self.img_pil)
        img_pil = tf.invert(img_tf)
        self.assertEqual(self.img_pil.size, img_pil.size)

    def test_repr(self):
        angle = 20
        center = (0, 0)
        tf = T.Rotation(angle=angle, center=center)
        self.assertIn('Rotation', repr(tf))
        self.assertIn(f'angle={angle}', repr(tf))
        self.assertIn(f'center={center}', repr(tf))

        tf2 = T.Rotation(angle=angle)
        self.assertNotIn(f'center', repr(tf2))

    def test_invert(self):
        tf = T.RandomRotation(degrees=180, expand=True)
        img_inv = tf(self.img_pil)
        tf_inv = tf.inverse()
        img = tf_inv(img_inv)
        self.assertEqual(img.size, self.img_pil.size)
        self.assertIsInstance(tf_inv, T.Rotation)
