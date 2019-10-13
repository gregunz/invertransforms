from torchvision import transforms

import invertransforms as T
from invertransforms.util import Invertible, flip_coin, InvertibleError


class Grayscale(transforms.Grayscale, Invertible):
    def inverse(self):
        return T.Lambda(
            lambd=lambda img: img,
            tf_inv=Grayscale(num_output_channels=self.num_output_channels),
            repr_str='GrayscaleInvert()'
        )


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
