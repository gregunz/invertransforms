import torch

import invertransforms as T
from invertransforms.util import Invertible
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
            self.tf.invert()

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
