"""
This module contains transformations for perspective transformation and flipping vertically or horizontally images.
These transformations can be applied deterministically or randomly.

"""
from PIL import Image
from torchvision import transforms

import invertransforms as T
from invertransforms import functional as F
from invertransforms.lib import InvertibleError, Invertible, flip_coin


class Perspective(Invertible):
    def __init__(self, startpoints, endpoints, interpolation=Image.BICUBIC):
        self.startpoints = startpoints
        self.endpoints = endpoints
        self.interpolation = interpolation

    def __call__(self, img):
        return F.perspective(img, self.startpoints, self.endpoints, self.interpolation)

    def inverse(self):
        return Perspective(
            startpoints=self.endpoints,
            endpoints=self.startpoints,
            interpolation=self.interpolation,
        )

    def __repr__(self):
        return f'{self.__class__.__name__}(startpoints={self.startpoints}, endpoints={self.endpoints})'


class RandomPerspective(transforms.RandomPerspective, Invertible):
    _transform = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be Perspectively transformed.

        Returns:
            PIL Image: Random perspectivley transformed image.
        """
        self._transform = T.Identity()
        if flip_coin(self.p):
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            self._transform = Perspective(
                startpoints=startpoints,
                endpoints=endpoints,
                interpolation=self.interpolation,
            )
        return self._transform(img)

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied.')

        return self._transform.inverse()

    def _can_invert(self):
        return self._transform is not None


class HorizontalFlip(Invertible):
    """
    Flip the image horizontally.
    """

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
    """
    Flip the image vertically.
    """

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
