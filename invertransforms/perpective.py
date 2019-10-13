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
    __startpoints = None
    __endpoints = None
    __do_tf = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be Perspectively transformed.

        Returns:
            PIL Image: Random perspectivley transformed image.
        """
        self.__do_tf = flip_coin(self.p)
        if self.__do_tf:
            width, height = img.size
            self.__startpoints, self.__endpoints = self.get_params(width, height, self.distortion_scale)
            return F.perspective(img, self.__startpoints, self.__endpoints, self.interpolation)
        return img

    def invert(self):
        if not self.__can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        if self.__do_tf:
            return Perspective(
                startpoints=self.__endpoints,
                endpoints=self.__startpoints,
                interpolation=self.interpolation,
            )
        else:
            return T.Identity()

    def __can_invert(self):
        return self.__do_tf is not None
