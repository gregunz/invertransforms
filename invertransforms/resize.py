import warnings

from torchvision import transforms

from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleException


class Resize(transforms.Resize, Invertible):
    img_h = img_w = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        self.img_w, self.img_h = img.size
        return super().__call__(img)

    def invert(self):
        if not self.__can_invert():
            raise InvertibleException('Cannot invert a transformation before it is applied'
                                      ' (size before resizing is unknown).')

        inverse = Resize(size=(self.img_h, self.img_w), interpolation=self.interpolation)
        inverse.img_h, inverse.tw = self.size
        return inverse

    def __can_invert(self):
        return self.img_w is not None and self.img_h is not None


class Scale(Resize):
    """
    Note: This transform is deprecated in favor of Resize.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the invertransforms.Scale transform is deprecated, " +
                      "please use invertransforms.Resize instead.")
        super().__init__(*args, **kwargs)
