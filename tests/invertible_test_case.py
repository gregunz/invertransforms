import random
import unittest

import torch

import invertransforms as T


class InvertibleTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.img_size = (256, 320)
        self.h, self.w = self.img_size
        self.crop_size = (64, 128)
        self.img_tensor = torch.randn((1,) + self.img_size).clamp(0, 1)

        self.img_pil = T.ToPILImage()(self.img_tensor)
        self.img_tensor = T.ToTensor()(self.img_pil)

        self.n = random.randint(0, 1e9)
