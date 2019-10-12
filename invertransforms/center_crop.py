from torchvision import transforms

from invertransforms import functional as F
from invertransforms.util import UndefinedInvertible


class CenterCrop(transforms.CenterCrop, UndefinedInvertible):
    img_h = img_w = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        self.img_w, self.img_h = img.size
        return F.center_crop(img, self.size)

    def _invert(self, **kwargs):
        inverse = CenterCrop(size=(self.img_h, self.img_h))  # center crop with bigger crop size is like padding
        th, tw = self.size
        inverse.img_w = tw
        inverse.img_h = th
        return inverse

    def _can_invert(self):
        return self.img_h is not None and self.img_w is not None
