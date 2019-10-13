from collections import Sequence

from torchvision import transforms

import invertransforms as T
from invertransforms import functional as F
from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleException


class Crop(Invertible):
    _img_h = _img_w = None

    def __init__(self, location, size):
        if isinstance(location, int):
            self.tl_i = self.tl_j = location
        elif isinstance(location, Sequence) and len(location) == 2:
            self.tl_i, self.tl_j = location
        else:
            raise Exception(f'Argument mismatch: location={location}')
        if isinstance(size, int):
            self.crop_h = self.crop_w = size
        elif isinstance(size, Sequence) and len(size) == 2:
            self.crop_h, self.crop_w = size
        else:
            raise Exception(f'Argument mismatch: size={size}')

    def __repr__(self):
        return f'{self.__class__.__name__}(location=({self.tl_i}, {self.tl_j}), size=({self.crop_h}, {self.crop_w}))'

    def __call__(self, img):
        self._img_w, self._img_h = img.size
        return F.crop(img, self.tl_i, self.tl_j, self.crop_h, self.crop_w)

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a transformation before it is applied'
                                      ' (size before cropping is unknown).')

        padding = (
            self.tl_j,
            self.tl_i,
            self._img_w - self.crop_w - self.tl_j,
            self._img_h - self.crop_h - self.tl_i,
        )
        inverse = T.Pad(padding=padding)
        inverse._img_w = self.crop_w
        inverse._img_h = self.crop_h
        return inverse

    def _can_invert(self):
        return self._img_w is not None and self._img_h is not None


class RandomCrop(transforms.RandomCrop, Invertible):
    img_h = img_w = tl_i = tl_j = None

    def get_params(self, img, output_size):
        self.img_w, self.img_h = img.size
        params = super().get_params(img, output_size)
        self.tl_i, self.tl_j, _, _ = params
        return params

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied')

        crop = Crop(
            location=(self.tl_i, self.tl_j),
            size=self.size,
        )
        crop._img_h, crop._img_w = self.img_h, self.img_w
        return crop.invert()

    def _can_invert(self):
        return self.img_h is not None or self.img_w is not None or self.tl_i is not None or self.tl_j is not None
