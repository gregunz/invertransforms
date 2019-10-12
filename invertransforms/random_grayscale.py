from torchvision import transforms

from invertransforms import Identity
from invertransforms.util import Invertible


class RandomGrayscale(transforms.RandomGrayscale, Invertible):
    def _invert(self, **kwargs):
        return Identity()
