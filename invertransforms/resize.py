import warnings

from torchvision import transforms

from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleException


class Resize(transforms.Resize, Invertible):
    _img_h = _img_w = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        self._img_w, self._img_h = img.size
        return super().__call__(img)

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a transformation before it is applied'
                                      ' (size before resizing is unknown).')

        inverse = Resize(size=(self._img_h, self._img_w), interpolation=self.interpolation)
        inverse._img_h, inverse.tw = self.size
        return inverse

    def _can_invert(self):
        return self._img_w is not None and self._img_h is not None


class Scale(Resize):
    """
    Note: This transform is deprecated in favor of Resize.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the invertransforms.Scale transform is deprecated, " +
                      "please use invertransforms.Resize instead.")
        super().__init__(*args, **kwargs)
