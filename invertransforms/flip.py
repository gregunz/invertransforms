from torchvision import transforms

import invertransforms as T
from invertransforms import functional as F
from invertransforms.util import Invertible, flip_coin, InvertibleError


class HorizontalFlip(Invertible):
    def __call__(self, img):
        return F.hflip(img)

    def inverse(self):
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

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied.')

        return self._transform.inverse()

    def _can_invert(self):
        return self._transform is not None


class VerticalFlip(Invertible):
    def __call__(self, img):
        return F.vflip(img)

    def inverse(self):
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

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied.')

        return self._transform.inverse()

    def _can_invert(self):
        return self._transform is not None