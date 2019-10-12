from collections import Sequence

from torchvision import transforms

from invertransforms import functional as F, FiveCrop, Lambda
from invertransforms.util import UndefinedInvertible


class TenCrop(transforms.TenCrop, UndefinedInvertible):
    five_crop = five_crop_flip = None

    def flip(self, img):
        if isinstance(img, Sequence):
            return [self.flip(im) for im in img]

        if self.vertical_flip:
            return F.vflip(img)
        else:
            return F.hflip(img)

    def __call__(self, img):
        self.five_crop = FiveCrop(self.size)
        self.five_crop_flip = FiveCrop(self.size)

        first_five = self.five_crop(img)
        second_five = self.five_crop_flip(self.flip(img))

        return first_five + second_five

    def _invert(self):
        five_crop = self.five_crop
        five_crop_flip = self.five_crop_flip
        return Lambda(
            lambd=lambda imgs: five_crop.invert()(imgs[:5]) + self.flip(five_crop_flip.invert()(imgs[5:])),
            lambd_inv=lambda imgs: five_crop(imgs[:5]) + five_crop_flip(self.flip(imgs[5:])),
            repr_str='TenCropInvert()',
        )

    def _can_invert(self):
        return self.five_crop is not None and self.five_crop_flip is not None
