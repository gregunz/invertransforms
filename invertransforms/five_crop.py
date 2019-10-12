from torchvision import transforms

from invertransforms import functional as F, Lambda, Pad
from invertransforms.util import UndefinedInvertible


class FiveCrop(transforms.FiveCrop, UndefinedInvertible):
    img_h = img_w = None

    def __call__(self, img):
        self.img_w, self.img_h = img.size
        return F.five_crop(img, self.size)

    def _invert(self, **kwargs):
        crop_h, crop_w = self.size
        pad_w = self.img_w - crop_w
        pad_h = self.img_h - crop_h
        pad_tl = (0, 0, pad_w, pad_h)
        pad_tr = (pad_w, 0, 0, pad_h)
        pad_bl = (0, pad_h, pad_w, 0)
        pad_br = (pad_w, pad_h, 0, 0)
        half_w = self.img_w // 2
        half_h = self.img_h // 2
        pad_center = (half_w, half_h, self.img_w - half_w, self.img_h - half_h)

        tfs = []
        for padding in [pad_tl, pad_tr, pad_bl, pad_br, pad_center]:
            tf = Pad(padding)
            tf.img_h = crop_h
            tf.img_w = crop_w
            tfs.append(tf)

        def invert_crops(tl, tr, bl, br, center):
            return [tf(img) for tf, img in zip(tfs, [tl, tr, bl, br, center])]

        def invert_invert_crops(tl, tr, bl, br, center):
            return [tf.invert()(img) for tf, img in zip(tfs, [tl, tr, bl, br, center])]

        return Lambda(
            lambd=invert_crops,
            lambd_inv=invert_invert_crops,
            repr_str='FiveCropInvert()',
        )

    def _can_invert(self):
        return self.img_w is not None and self.img_h is not None
