import torch

import invertransforms as T
from invertransforms.lib import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestLambda(InvertibleTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tf = T.Lambda(
            lambd=lambda x: x * 2,
            tf_inv=lambda x: x * 0.5
        )

    def test_call(self):
        tf = T.Lambda(lambd=lambda x: self.n)
        self.assertEqual(tf(None), self.n)

    def test_invert_lambda(self):
        self.assertTrue(torch.allclose(self.img_tensor, self.tf.invert(self.tf(self.img_tensor))))

    def test_invert_invertible(self):
        tf_inv = T.Lambda(lambda x: x * 0.5)
        tf = T.Lambda(
            lambd=lambda x: x * 2,
            tf_inv=tf_inv
        )
        self.assertEqual(tf_inv, tf.inverse())
        self.assertTrue(torch.allclose(self.img_tensor, tf.invert(tf(self.img_tensor))))

    def test_repr(self):
        tf_inv = self.tf.inverse()
        self.assertIn('Inverse()', repr(tf_inv))
        tf_inv_inv = tf_inv.inverse()
        self.assertNotIn('Inverse()', repr(tf_inv_inv))

    def test_invert_none(self):
        tf = T.Lambda(lambda x: self.n)
        self.assertEqual(self.n, tf(None))
        with self.assertRaises(InvertibleError):
            tf.inverse()
