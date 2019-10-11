import random

from torchvision import transforms

from transforms import Identity, Compose
from transforms.util import UndefinedInvertible


class RandomApply(transforms.RandomApply, UndefinedInvertible):
    no_tf = None

    def __call__(self, img):
        self.no_tf = self.p < random.random()
        if self.no_tf:
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def _invert(self):
        if self.no_tf:
            return Identity()
        else:
            return Compose(self.transforms).invert()

    def _can_invert(self):
        return self.no_tf is not None
