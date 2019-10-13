import torch

import invertransforms as T
from invertransforms.util import InvertibleException
from tests.invertible_test_case import InvertibleTestCase


class TestLambda(InvertibleTestCase):

    def test_call(self):
        tf = T.Lambda(lambd=lambda x: self.n)
        self.assertEqual(tf(None), self.n)

    def test_invert_lambda(self):
        tf = T.Lambda(
            lambd=lambda x: x * 2,
            tf_inv=lambda x: x * 0.5
        )
        self.assertTrue(torch.allclose(self.img_tensor, tf.inverse(tf(self.img_tensor))))

    def test_invert_invertible(self):
        tf_inv = T.Lambda(lambda x: x * 0.5)
        tf = T.Lambda(
            lambd=lambda x: x * 2,
            tf_inv=tf_inv
        )
        self.assertEqual(tf_inv, tf.invert())
        self.assertTrue(torch.allclose(self.img_tensor, tf.inverse(tf(self.img_tensor))))

    def test_repr(self):
        tf = T.Lambda(
            lambd=lambda x: x,
            tf_inv=lambda x: x,
        )
        tf_inv = tf.invert()
        self.assertIn('Invert()', repr(tf_inv))
        tf_inv_inv = tf_inv.invert()
        self.assertNotIn('Invert()', repr(tf_inv_inv))

    def test_invert_none(self):
        with self.assertRaises(InvertibleException):
            T.Lambda(lambda: ()).invert()
