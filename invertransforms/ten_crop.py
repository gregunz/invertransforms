from collections import Sequence

from torchvision import transforms

import invertransforms as T
from invertransforms import functional as F
from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleException


class TenCrop(transforms.TenCrop, Invertible):
    _five_crop = _five_crop_flip = None

    def flip(self, img):
        if isinstance(img, Sequence):
            return [self.flip(im) for im in img]

        if self.vertical_flip:
            return F.vflip(img)
        else:
            return F.hflip(img)

    def __call__(self, img):
        self._five_crop = T.FiveCrop(self.size)
        self._five_crop_flip = T.FiveCrop(self.size)

        first_five = self._five_crop(img)
        second_five = self._five_crop_flip(self.flip(img))

        return first_five + second_five

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a transformation before it is applied'
                                      ' (size before cropping is unknown).')

        five_crop = self._five_crop
        five_crop_flip = self._five_crop_flip
        return T.Lambda(
            lambd=lambda imgs: five_crop.invert()(imgs[:5]) + self.flip(five_crop_flip.invert()(imgs[5:])),
            tf_inv=lambda imgs: five_crop.invert().inverse(imgs[:5]) + five_crop_flip.invert().inverse(
                self.flip(imgs[5:])),
            repr_str='TenCropInvert()',
        )

    def _can_invert(self):
        return self._five_crop is not None and self._five_crop_flip is not None
