from torchvision import transforms

from invertransforms import functional as F
from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleException


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

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a transformation before it is applied'
                                      ' (size before cropping is unknown).')

        inverse = CenterCrop(size=(self._img_h, self._img_w))  # center crop with bigger crop size is like padding
        th, tw = self.size
        inverse._img_w = tw
        inverse._img_h = th
        return inverse

    def _can_invert(self):
        return self._img_h is not None and self._img_w is not None
