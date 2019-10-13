from torchvision import transforms

import invertransforms as T
from invertransforms import functional as F


class RandomHorizontalFlip(T.RandomLambda, transforms.RandomHorizontalFlip):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__(lambd=F.hflip, tf_inv=F.hflip, p=p, repr_str='HorizontalFlip()')
