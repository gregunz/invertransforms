import torch
from PIL import Image

import invertransforms as T
from invertransforms.lib import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestCompose(InvertibleTestCase):

    def test_invert_of_invertible_only(self):
        tf = T.Compose(['s'])
        with self.assertRaises(InvertibleError):
            tf.inverse()

    def test_nested_invert(self):
        tf = T.Compose([
            T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(0),
                T.RandomHorizontalFlip(1),
            ]),
        ])
        img_pil = tf(self.img_tensor)
        self.assertIsInstance(img_pil, Image.Image)
        self.assertIsInstance(tf.invert(img_pil), torch.Tensor)
