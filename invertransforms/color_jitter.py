from torchvision import transforms

from invertransforms import Identity
from invertransforms.util import Invertible


class ColorJitter(transforms.ColorJitter, Invertible):
    def _invert(self, **kwargs):
        return Identity()
