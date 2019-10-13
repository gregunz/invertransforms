import torch

import invertransforms as T
from invertransforms.util import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class Test(InvertibleTestCase):

    def test_invert(self):
        small_img_tensor = torch.randn(1, 16, 16)

        tf = T.LinearTransformation(
            transformation_matrix=torch.eye(small_img_tensor.nelement()) * 2,
            mean_vector=torch.randn_like(small_img_tensor).view(-1)
        )
        self.assertTrue(torch.allclose(tf.invert(tf(small_img_tensor)), small_img_tensor, atol=1e-07))

    def test_not_invertible(self):
        with self.assertRaises(InvertibleError):
            T.LinearTransformation(
                transformation_matrix=torch.zeros((32, 32)),
                mean_vector=torch.randn(32),
            ).inverse()
