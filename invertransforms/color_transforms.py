from torchvision import transforms
from torchvision.transforms import functional as F

import invertransforms as T
from invertransforms.util import Invertible, flip_coin, InvertibleError
from invertransforms.util_functions import Identity


class ColorJitter(transforms.ColorJitter, Invertible):
    def inverse(self):
        return Identity()


class Grayscale(transforms.Grayscale, Invertible):
    def __call__(self, img):
        return F.to_grayscale(img, num_output_channels=self.num_output_channels)

    def inverse(self):
        return T.Identity()


class RandomGrayscale(transforms.RandomGrayscale, Invertible):
    _transform = None

    def __call__(self, img):
        self._transform = T.Identity()
        if flip_coin(self.p):
            num_output_channels = 1 if img.mode == 'L' else 3
            self._transform = Grayscale(num_output_channels=num_output_channels)
        return self._transform(img)

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied.')

        return self._transform.inverse()

    def _can_invert(self):
        return self._transform is not None
