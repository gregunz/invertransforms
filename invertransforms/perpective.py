from PIL import Image
from torchvision import transforms

import invertransforms as T
from invertransforms import functional as F
from invertransforms.util import Invertible, flip_coin
from invertransforms.util.invertible import InvertibleException


class Perspective(Invertible):
    def __init__(self, startpoints, endpoints, interpolation=Image.BICUBIC):
        self.startpoints = startpoints
        self.endpoints = endpoints
        self.interpolation = interpolation

    def __call__(self, img):
        return F.perspective(img, self.startpoints, self.endpoints, self.interpolation)

    def invert(self):
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

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        return self._transform.invert()

    def _can_invert(self):
        return self._transform is not None
