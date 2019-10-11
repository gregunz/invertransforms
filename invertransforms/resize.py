import warnings

from torchvision import transforms

from invertransforms.util import UndefinedInvertible


class Resize(transforms.Resize, UndefinedInvertible):
    img_w = img_h = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        self.img_w, self.img_h = img.size
        return super().__call__(img)

    def _invert(self):
        return Resize(size=(self.img_h, self.img_w), interpolation=self.interpolation)

    def _can_invert(self):
        return self.img_w is not None and self.img_h is not None


class Scale(Resize):
    """
    Note: This transform is deprecated in favor of Resize.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the invertransforms.Scale transform is deprecated, " +
                      "please use invertransforms.Resize instead.")
        super().__init__(*args, **kwargs)
