import random

from torchvision import transforms

from transforms.util import UndefinedInvertible, _Invertible


class RandomApply(transforms.RandomApply, UndefinedInvertible):
    no_tf = None

    def __call__(self, img):
        self.no_tf = self.p < random.random()
        if self.no_tf:
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def _invert(self, **kwargs):
        if self.no_tf:
            return
        transforms_inv = []
        for t in self.transforms[::-1]:
            assert isinstance(t, _Invertible), f'{t.__class__.__name__} is not an invertible class'
            transforms_inv.append(t.invert())
        return

    def _can_invert(self):
        return self.no_tf is not None
