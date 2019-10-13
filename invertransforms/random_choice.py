import random

from torchvision import transforms

from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleException


class RandomChoice(transforms.RandomChoice, Invertible):
    transform = None

    def __call__(self, img):
        i = random.randint(0, len(self.transforms) - 1)
        self.transform = self.transforms[i]
        return self.transform(img)

    def invert(self):
        if not self.__can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        return self.transform.invert()

    def __can_invert(self):
        return self.transform is not None
