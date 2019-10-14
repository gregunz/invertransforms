"""
Crop and Pad module.

This modules contains multiple transformations about creating crops.
Generally, their inverse is/or involves `Pad`, and respectively is `Crop` for `Pad` transformation.

"""
from typing import Sequence

from torchvision import transforms

import invertransforms as T
from invertransforms import functional as F
from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleError


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

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a transformation before it is applied'
                                  ' (size before cropping is unknown).')

        padding = (
            self.tl_j,
            self.tl_i,
            self._img_w - self.crop_w - self.tl_j,
            self._img_h - self.crop_h - self.tl_i,
        )
        inverse = Pad(padding=padding)
        inverse._img_w = self.crop_w
        inverse._img_h = self.crop_h
        return inverse

    def _can_invert(self):
        return self._img_w is not None and self._img_h is not None


class CenterCrop(transforms.CenterCrop, Invertible):
    _img_h = _img_w = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        self._img_w, self._img_h = img.size
        return F.center_crop(img, self.size)

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a transformation before it is applied'
                                  ' (size before cropping is unknown).')

        inverse = CenterCrop(size=(self._img_h, self._img_w))  # center crop with bigger crop size is like padding
        th, tw = self.size
        inverse._img_w = tw
        inverse._img_h = th
        return inverse

    def _can_invert(self):
        return self._img_h is not None and self._img_w is not None


class RandomCrop(transforms.RandomCrop, Invertible):
    _img_h = _img_w = _tl_i = _tl_j = None

    def get_params(self, img, output_size):
        self._img_w, self._img_h = img.size
        params = super().get_params(img, output_size)
        self._tl_i, self._tl_j, _, _ = params
        return params

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied')

        crop = Crop(
            location=(self._tl_i, self._tl_j),
            size=self.size,
        )
        crop._img_h, crop._img_w = self._img_h, self._img_w
        return crop.inverse()

    def _can_invert(self):
        return self._img_h is not None or self._img_w is not None or self._tl_i is not None or self._tl_j is not None


class Pad(transforms.Pad, Invertible):
    _img_h = _img_w = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        self._img_w, self._img_h = img.size
        return super().__call__(img=img)

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a transformation before it is applied'
                                  ' (size of image before padding unknown).')

        padding = self.padding
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        elif isinstance(padding, Sequence) and len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        elif isinstance(padding, Sequence) and len(padding) == 4:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]
        else:
            raise Exception(f'Argument mismatch: padding={padding}')

        size = (self._img_h, self._img_w)
        location = (pad_top, pad_left)
        inverse = Crop(location=location, size=size)
        inverse._img_h = pad_top + self._img_h + pad_bottom
        inverse._img_w = pad_left + self._img_w + pad_right
        return inverse

    def _can_invert(self):
        return self._img_w is not None and self._img_h is not None


class FiveCrop(transforms.FiveCrop, Invertible):
    _img_h = _img_w = None

    def __call__(self, img):
        self._img_w, self._img_h = img.size
        return F.five_crop(img, self.size)

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a transformation before it is applied'
                                  ' (size before the cropping is unknown).')

        crop_h, crop_w = self.size
        pad_w = self._img_w - crop_w
        pad_h = self._img_h - crop_h
        pad_tl = (0, 0, pad_w, pad_h)
        pad_tr = (pad_w, 0, 0, pad_h)
        pad_bl = (0, pad_h, pad_w, 0)
        pad_br = (pad_w, pad_h, 0, 0)
        half_w = (self._img_w - crop_w) // 2
        half_h = (self._img_h - crop_h) // 2
        pad_center = (half_w, half_h, self._img_w - crop_w - half_w, self._img_h - crop_h - half_h)

        tfs = []
        for padding in [pad_tl, pad_tr, pad_bl, pad_br, pad_center]:
            tf = T.Pad(padding)
            tf._img_h = crop_h
            tf._img_w = crop_w
            tfs.append(tf)

        def invert_crops(crops):
            tl, tr, bl, br, center = crops
            return [tf(img) for tf, img in zip(tfs, [tl, tr, bl, br, center])]

        def invert_invert_crops(crops):
            tl, tr, bl, br, center = crops
            return [tf.inverse()(img) for tf, img in zip(tfs, [tl, tr, bl, br, center])]

        return T.Lambda(
            lambd=invert_crops,
            tf_inv=invert_invert_crops,
            repr_str=f'FiveCropInverse()',
        )

    def _can_invert(self):
        return self._img_w is not None and self._img_h is not None


class TenCrop(transforms.TenCrop, Invertible):
    _five_crop = _five_crop_flip = None

    def _flip(self, img):
        if isinstance(img, Sequence):
            return [self._flip(im) for im in img]

        if self.vertical_flip:
            return F.vflip(img)
        else:
            return F.hflip(img)

    def __call__(self, img):
        self._five_crop = FiveCrop(self.size)
        self._five_crop_flip = FiveCrop(self.size)

        first_five = self._five_crop(img)
        second_five = self._five_crop_flip(self._flip(img))

        return first_five + second_five

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a transformation before it is applied'
                                  ' (size before cropping is unknown).')

        five_crop = self._five_crop
        five_crop_flip = self._five_crop_flip
        return T.Lambda(
            lambd=lambda imgs: five_crop.inverse()(imgs[:5]) + self._flip(five_crop_flip.inverse()(imgs[5:])),
            tf_inv=lambda imgs: five_crop.inverse().invert(imgs[:5]) + five_crop_flip.inverse().invert(
                self._flip(imgs[5:])),
            repr_str='TenCropInverse()',
        )

    def _can_invert(self):
        return self._five_crop is not None and self._five_crop_flip is not None
