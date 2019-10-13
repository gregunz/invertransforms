from torchvision import transforms

from invertransforms.identity import Identity
from invertransforms.util import Invertible


class ColorJitter(transforms.ColorJitter, Invertible):
    def invert(self):
        return Identity()
