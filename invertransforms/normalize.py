import torch
from torchvision import transforms

from invertransforms.util import Invertible


class Normalize(transforms.Normalize, Invertible):
    def invert(self):
        mean = torch.as_tensor(self.mean)
        std = torch.as_tensor(self.std)
        std_inv = torch.tensor(1.0) / std
        mean_inv = (-1.0) * mean * std_inv
        return Normalize(mean=mean_inv, std=std_inv, inplace=self.inplace)
