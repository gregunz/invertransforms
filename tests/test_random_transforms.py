import torch
from PIL import Image

import invertransforms as T
import invertransforms.crop_pad
import invertransforms.util_functions
from invertransforms.util import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestRandomApply(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleError):
            T.RandomApply([]).inverse()

    def test_invert(self):
        tf = T.RandomApply(
            transforms=[T.Identity(), invertransforms.util_functions.ToTensor()],
            p=1,
        )

        img_tensor = tf(self.img_pil)
        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertIsInstance(tf.invert(img_tensor), Image.Image)

    def test_identity(self):
        tf_id = T.RandomApply(
            transforms=invertransforms.util_functions.ToPILImage(),
            p=0,
        )

        self.assertEqual(tf_id(self.n), self.n)
        self.assertEqual(tf_id.invert(self.n), self.n)


class TestRandomChoice(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleError):
            T.RandomChoice([]).inverse()

    def test_invert(self):
        tf = T.RandomChoice(
            [invertransforms.util_functions.ToTensor()] * 10
        )
        self.assertIsInstance(tf(self.img_pil), torch.Tensor)
        self.assertIsInstance(tf.inverse(), invertransforms.util_functions.ToPILImage)


class TestRandomOrder(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleError):
            T.RandomOrder([]).inverse()

    def test_invert(self):
        tf = T.RandomOrder([
            invertransforms.crop_pad.CenterCrop(self.crop_size),
            T.RandomHorizontalFlip(1),
            T.RandomVerticalFlip(1),
        ])
        img_inv = tf(self.img_pil)
        img_pil = tf.invert(img_inv)
        self.assertEqual(img_pil.size, self.img_pil.size)
        center = (self.w // 2 - 10, self.h // 2 + 10)
        self.assertEqual(img_pil.getpixel(center), self.img_pil.getpixel(center))
