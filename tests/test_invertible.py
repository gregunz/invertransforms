import torch

import invertransforms as T
from invertransforms.lib import Invertible
from tests.invertible_test_case import InvertibleTestCase


class TestInvertible(InvertibleTestCase):
    def setUp(self) -> None:
        super().setUp()

        # This test is used for accessing abstract methods
        class NewInvertible(Invertible):
            pass

        self.tf = NewInvertible()

    def test_access_abstract_methods(self):
        with self.assertRaises(NotImplementedError):
            self.tf(None)

        with self.assertRaises(NotImplementedError):
            self.tf.inverse()

        # self.assertTrue(self.tf._can_invert())

    def test_replay(self):
        tf = T.Compose([
            T.RandomCrop(self.crop_size),
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        img_tf1 = tf.replay(self.img_pil)
        img_tf2 = tf.replay(self.img_pil)
        self.assertTrue(torch.allclose(img_tf1, img_tf2))

    def test_repr(self):
        self.assertIn(self.tf.__class__.__name__, repr(self.tf))

    def test_track(self):
        tf = T.Compose([
            T.ToPILImage(),
            T.RandomVerticalFlip(p=0.5),
            # the crop will include the center pixels
            T.RandomCrop(size=tuple(int(0.8 * self.img_size[i]) for i in range(2))),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
        ])

        imgs_tf = [tf.track(self.img_tensor) for _ in range(10)]
        for i, img_tf in enumerate(imgs_tf):
            n = min(self.img_size) // 10
            center_pixels = (0,) + tuple(slice(self.img_size[i] // 2 - n, self.img_size[i] // 2 + n) for i in range(2))
            self.assertTrue(torch.allclose(tf[i](img_tf)[center_pixels],
                                           self.img_tensor[center_pixels]))
