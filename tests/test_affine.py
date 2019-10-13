import invertransforms as T
from invertransforms.util import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class Test(InvertibleTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.degrees = [0, 180]
        self.tf = T.RandomAffine(
            degrees=self.degrees,
            translate=(0.5, 0.8),
            scale=(1.5, 0.5),
        )

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleError):
            self.tf.inverse()

    def test_invert(self):
        img_inv = self.tf(self.img_pil)
        img_pil = self.tf.invert(img_inv)
        # could find a better way to be sure affine transformation gives back the right image
        self.assertEqual(img_pil.size, self.img_pil.size)

    def test_invert_invert(self):
        img_inv = self.tf(self.img_pil)
        tf_inv = self.tf.inverse()
        img_pil = tf_inv(img_inv)
        img_inv2 = tf_inv.invert(img_pil)
        self.assertEqual(img_inv.size, img_inv2.size)

    def test_repr(self):
        self.tf(self.img_pil)
        affine = self.tf.inverse()
        self.assertIn('Affine', repr(affine))
        self.assertIn('matrix=', repr(affine))
