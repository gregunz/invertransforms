import random
import unittest

import torch
from torchvision.transforms import ToPILImage


class InvertibleTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.img_size = (256, 320)
        self.h, self.w = self.img_size
        self.crop_size = (64, 128)
        self.img_tensor = torch.randn((1,) + self.img_size).clamp(0, 1)
        self.img_pil = ToPILImage()(self.img_tensor)

        self.n = random.randint(0, 1e9)
