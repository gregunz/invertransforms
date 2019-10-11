from torchvision import transforms
from torchvision.transforms import functional as F

from invertransforms import Identity, Lambda
from invertransforms.util import UndefinedInvertible, flip_coin


class VerticalFlip(Lambda):
    def __init__(self):
        super().__init__(
            lambd=F.vflip,
            lambd_inv=F.vflip,
        )


class RandomVerticalFlip(transforms.RandomVerticalFlip, UndefinedInvertible):
    do_tf = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        self.do_tf = flip_coin(self.p)
        if self.do_tf:
            return F.vflip(img)
        return img

    def _invert(self):
        if self.do_tf:
            return VerticalFlip()  # such that we have nice __repr__ (could have been a Lambda)
        else:
            return Identity()

    def _can_invert(self):
        return self.do_tf is not None
