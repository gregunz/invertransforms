import torch
from torchvision import transforms

from invertransforms.util import Invertible


class Normalize(transforms.Normalize, Invertible):
    def _invert(self):
        mean = torch.as_tensor(self.mean)
        std = torch.as_tensor(self.std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = (-1) * mean * std_inv
        return Normalize(mean=mean_inv, std=std_inv, inplace=self.inplace)
