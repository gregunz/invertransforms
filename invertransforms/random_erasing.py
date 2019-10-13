from torchvision import transforms

from invertransforms.identity import Identity
from invertransforms.util import Invertible


class RandomErasing(transforms.RandomErasing, Invertible):
    def invert(self):
        return Identity()
