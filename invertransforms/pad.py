from typing import Sequence

from torchvision import transforms

import invertransforms as T
from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleException


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

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a transformation before it is applied'
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
        inverse = T.Crop(location=location, size=size)
        inverse._img_h = pad_top + self._img_h + pad_bottom
        inverse._img_w = pad_left + self._img_w + pad_right
        return inverse

    def _can_invert(self):
        return self._img_w is not None and self._img_h is not None
