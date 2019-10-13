import torch

import invertransforms.tensor_transforms
from tests.invertible_test_case import InvertibleTestCase


class TestNormalize(InvertibleTestCase):

    def test_invert(self):
        tf = invertransforms.tensor_transforms.Normalize(mean=[0.5], std=[0.5])
        self.assertTrue(torch.allclose(self.img_tensor, tf.invert(tf(self.img_tensor)), atol=1e-07))
