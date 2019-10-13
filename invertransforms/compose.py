from torchvision import transforms

from invertransforms.util import Invertible, InvertibleException


class Compose(transforms.Compose, Invertible):
    def invert(self):
        transforms_inv = []
        for t in self.transforms[::-1]:
            if not isinstance(t, Invertible):
                raise InvertibleException(f'{t} ({t.__class__.__name__}) is not an invertible object')
            transforms_inv.append(t.invert())
        return Compose(transforms=transforms_inv)
