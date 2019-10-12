from torchvision import transforms

from invertransforms import Pad
from invertransforms.util import UndefinedInvertible


class RandomCrop(transforms.RandomCrop, UndefinedInvertible):
    img_h = img_w = tl_i = tl_j = None

    def get_params(self, img, output_size):
        self.img_w, self.img_h = img.size
        params = super().get_params(img, output_size)
        self.tl_i, self.tl_j, _, _ = params
        return params

    def _invert(self):
        crop_h, crop_w = self.size
        padding = (
            self.tl_j,
            self.tl_i,
            self.img_w - crop_w - self.tl_j,
            self.img_h - crop_h - self.tl_i,
        )
        inverse = Pad(padding=padding)
        inverse.img_h = crop_h
        inverse.img_w = crop_w
        return inverse

    def _can_invert(self):
        return self.img_h is None or self.img_w is None or self.tl_i is None or self.tl_j is None
