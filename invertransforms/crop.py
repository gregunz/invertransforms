from collections import Sequence

from invertransforms import functional as F
from invertransforms.pad import Pad
from invertransforms.util import UndefinedInvertible


class Crop(UndefinedInvertible):
    img_h = img_w = None

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
        return f'{self.__class__.__name__}(location=({self.tl_i},{self.tl_j}), size=({self.crop_h},{self.crop_w}))'

    def __call__(self, img):
        self.img_w, self.img_h = img.size
        return F.crop(img, self.tl_i, self.tl_j, self.crop_h, self.crop_w)

    def _invert(self, **kwargs):
        padding = (
            self.tl_j,
            self.tl_i,
            self.img_w - self.crop_w - self.tl_j,
            self.img_h - self.crop_h - self.tl_i,
        )
        inverse = Pad(padding=padding, **kwargs)
        inverse.img_w = self.crop_w
        inverse.img_h = self.crop_h
        return inverse

    def _can_invert(self):
        return self.img_w is not None and self.img_h is not None
