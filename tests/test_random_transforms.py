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
            transforms=T.Lambda(lambda x: 2 * x),
            p=0,
        )

        self.assertEqual(tf_id(self.n), self.n)
        self.assertEqual(tf_id.inverse(self.n), self.n)


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


class TestRandomOrder(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleException):
            T.RandomOrder([]).invert()

    def test_invert(self):
        tf = T.RandomOrder([
            T.CenterCrop(self.crop_size),
            T.RandomHorizontalFlip(1),
            T.RandomVerticalFlip(1),
        ])
        img_inv = tf(self.img_pil)
        img_pil = tf.inverse(img_inv)
        self.assertEqual(img_pil.size, self.img_pil.size)
        center = (self.w // 2 - 10, self.h // 2 + 10)
        self.assertEqual(img_pil.getpixel(center), self.img_pil.getpixel(center))
