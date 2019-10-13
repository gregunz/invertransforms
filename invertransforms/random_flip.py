from torchvision import transforms

import invertransforms as T
from invertransforms import functional as F
from invertransforms.util import Invertible, flip_coin, InvertibleException


class HorizontalFlip(Invertible):
    def __call__(self, img):
        return F.hflip(img)

    def invert(self):
        return HorizontalFlip()


class RandomHorizontalFlip(transforms.RandomHorizontalFlip, Invertible):
    _transform = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        self._transform = T.Identity()
        if flip_coin(self.p):
            self._transform = HorizontalFlip()
        return self._transform(img)

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        return self._transform.invert()

    def _can_invert(self):
        return self._transform is not None


class VerticalFlip(Invertible):
    def __call__(self, img):
        return F.vflip(img)

    def invert(self):
        return VerticalFlip()


class RandomVerticalFlip(transforms.RandomVerticalFlip, Invertible):
    _transform = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        self._transform = T.Identity()
        if flip_coin(self.p):
            self._transform = VerticalFlip()
        return self._transform(img)

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        return self._transform.invert()

    def _can_invert(self):
        return self._transform is not None
