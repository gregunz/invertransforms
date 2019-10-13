import torch
from PIL import Image

import invertransforms as T
import invertransforms.list_transforms
import invertransforms.util_functions
from invertransforms.util import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestCompose(InvertibleTestCase):

    def test_invert_of_invertible_only(self):
        tf = invertransforms.list_transforms.Compose(['s'])
        with self.assertRaises(InvertibleError):
            tf.inverse()

    def test_nested_invert(self):
        tf = invertransforms.list_transforms.Compose([
            invertransforms.list_transforms.Compose([
                invertransforms.util_functions.ToPILImage(),
                T.RandomHorizontalFlip(0),
                T.RandomHorizontalFlip(1),
            ]),
        ])
        img_pil = tf(self.img_tensor)
        self.assertIsInstance(img_pil, Image.Image)
        self.assertIsInstance(tf.invert(img_pil), torch.Tensor)
