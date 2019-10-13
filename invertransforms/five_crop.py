from torchvision import transforms

import invertransforms as T
from invertransforms import functional as F
from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleError


class FiveCrop(transforms.FiveCrop, Invertible):
    _img_h = _img_w = None

    def __call__(self, img):
        self._img_w, self._img_h = img.size
        return F.five_crop(img, self.size)

    def invert(self):
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
            return [tf.invert()(img) for tf, img in zip(tfs, [tl, tr, bl, br, center])]

        return T.Lambda(
            lambd=invert_crops,
            tf_inv=invert_invert_crops,
            repr_str=f'FiveCropInvert()',
        )

    def _can_invert(self):
        return self._img_w is not None and self._img_h is not None
