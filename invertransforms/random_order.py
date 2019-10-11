import random

from torchvision import transforms

from invertransforms import Compose
from invertransforms.util import UndefinedInvertible


class RandomOrder(transforms.RandomOrder, UndefinedInvertible):
    order = None

    def __call__(self, img):
        self.order = list(range(len(self.transforms)))
        random.shuffle(self.order)
        for i in self.order:
            img = self.transforms[i](img)
        return img

    def _invert(self):
        return Compose(transforms=[self.transforms[i].invert() for i in self.order[::-1]])

    def _can_invert(self):
        return self.order is not None
