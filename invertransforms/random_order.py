import random

from torchvision import transforms

from invertransforms.compose import Compose
from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleException


class RandomOrder(transforms.RandomOrder, Invertible):
    order = None

    def __call__(self, img):
        self.order = list(range(len(self.transforms)))
        random.shuffle(self.order)
        for i in self.order:
            img = self.transforms[i](img)
        return img

    def invert(self):
        if not self.__can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        return Compose(transforms=[self.transforms[i].invert() for i in self.order[::-1]])

    def __can_invert(self):
        return self.order is not None
