"""
This modules contains transformations which resize images.
"""

import warnings

from torchvision import transforms

import invertransforms as T
from invertransforms.lib import InvertibleError, Invertible


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

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a transformation before it is applied'
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


class RandomResizedCrop(transforms.RandomResizedCrop, Invertible):
    _transform = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = super().get_params(img, self.scale, self.ratio)
        self._transform = T.Compose([
            T.Crop(location=(i, j), size=(h, w)),
            T.Resize(size=self.size, interpolation=self.interpolation),
        ])
        return self._transform(img)

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied.')

        return self._transform.inverse()

    def _can_invert(self):
        return self._transform is not None


class RandomSizedCrop(RandomResizedCrop):
    """
    Note: This transform is deprecated in favor of RandomResizedCrop.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.RandomSizedCrop transform is deprecated, " +
                      "please use transforms.RandomResizedCrop instead.")
        super(RandomSizedCrop, self).__init__(*args, **kwargs)
