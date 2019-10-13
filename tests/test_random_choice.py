import torch

import invertransforms as T
from invertransforms.util import InvertibleException
from tests.invertible_test_case import InvertibleTestCase


class TestRandomChoice(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleException):
            T.RandomChoice([]).invert()

    def test_invert(self):
        tf = T.RandomChoice(
            [T.ToTensor()] * 10
        )
        self.assertIsInstance(tf(self.img_pil), torch.Tensor)
        self.assertIsInstance(tf.invert(), T.ToPILImage)
