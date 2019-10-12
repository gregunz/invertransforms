from torchvision import transforms

from invertransforms import functional as F
from invertransforms.random_lambda import RandomLambda


class RandomHorizontalFlip(RandomLambda, transforms.RandomHorizontalFlip):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p):
        super().__init__(lambd=F.hflip, lambd_inv=F.hflip, p=p, repr_str='HorizontalFlip()')
