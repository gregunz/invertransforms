import warnings

from torchvision import transforms

import invertransforms as T
import invertransforms.list_transforms
from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleError


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
        self._transform = invertransforms.list_transforms.Compose([
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
