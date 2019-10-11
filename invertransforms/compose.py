from torchvision import transforms

from invertransforms.util import Invertible, _Invertible


class Compose(transforms.Compose, Invertible):
    def _invert(self):
        transforms_inv = []
        for t in self.transforms[::-1]:
            assert isinstance(t, _Invertible), f'{t.__class__.__name__} is not an invertible class'
            transforms_inv.append(t.invert())
        return Compose(transforms=transforms_inv)
