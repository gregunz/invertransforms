import random
from typing import Union

from torchvision import transforms

import invertransforms as T
from invertransforms.util import Invertible, flip_coin
from invertransforms.util.invertible import InvertibleException


class RandomApply(transforms.RandomApply, Invertible):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms: one or multiple of transformations
        p (float): probability
    """

    def __init__(self, transforms: Union[Invertible, list, tuple], p=0.5):
        if isinstance(transforms, Invertible):
            transforms = [transforms]
        super().__init__(transforms=transforms, p=0.5)
        self.p = p
        self._transform = None

    def __call__(self, img):

        self._transform = T.Identity()
        if flip_coin(self.p):
            self._transform = T.Compose(self.transforms)
        return self._transform(img)

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        return self._transform.invert()

    def _can_invert(self):
        return self._transform is not None


class RandomChoice(transforms.RandomChoice, Invertible):
    _transform = None

    def __call__(self, img):
        i = random.randint(0, len(self.transforms) - 1)
        self._transform = self.transforms[i]
        return self._transform(img)

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        return self._transform.invert()

    def _can_invert(self):
        return self._transform is not None


class RandomOrder(transforms.RandomOrder, Invertible):
    _order = None

    def __call__(self, img):
        self._order = list(range(len(self.transforms)))
        random.shuffle(self._order)
        for i in self._order:
            img = self.transforms[i](img)
        return img

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        return T.Compose(transforms=[self.transforms[i].invert() for i in self._order[::-1]])

    def _can_invert(self):
        return self._order is not None
