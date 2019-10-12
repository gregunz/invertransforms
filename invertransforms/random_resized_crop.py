import warnings

from torchvision import transforms

from invertransforms import Crop, Resize, Compose
from invertransforms.util import UndefinedInvertible


class RandomResizedCrop(transforms.RandomResizedCrop, UndefinedInvertible):
    tf = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = super().get_params(img, self.scale, self.ratio)
        self.tf = Compose([
            Crop(location=(i, j), size=(h, w)),
            Resize(size=self.size, interpolation=self.interpolation),
        ])
        return self.tf(img)

    def _invert(self, **kwargs):
        return self.tf.invert()

    def _can_invert(self):
        return self.tf is not None


class RandomSizedCrop(RandomResizedCrop):
    """
    Note: This transform is deprecated in favor of RandomResizedCrop.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.RandomSizedCrop transform is deprecated, " +
                      "please use transforms.RandomResizedCrop instead.")
        super(RandomSizedCrop, self).__init__(*args, **kwargs)
