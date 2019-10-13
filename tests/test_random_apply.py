import torch
from PIL import Image

import invertransforms as T
from invertransforms.util import InvertibleException
from tests.invertible_test_case import InvertibleTestCase


class TestRandomApply(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleException):
            T.RandomApply([]).invert()

    def test_invert(self):
        tf = T.RandomApply(
            transforms=[T.Identity(), T.ToTensor()],
            p=1,
        )

        img_tensor = tf(self.img_pil)
        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertIsInstance(tf.inverse(img_tensor), Image.Image)

    def test_identity(self):
        tf_id = T.RandomApply(
            transforms=[T.Identity(), T.ToTensor()],
            p=0,
        )

        self.assertEqual(tf_id(self.n), self.n)
        self.assertEqual(tf_id.inverse(self.n), self.n)
