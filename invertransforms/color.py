"""
This modules contains transformations on the image channels (RGB, grayscale).

Technically these transformations cannot be inverted or it simply makes not much sense,
hence the inverse is usually the identity function.
"""
from torchvision import transforms
from torchvision.transforms import functional as F

import invertransforms as T
from invertransforms.lib import InvertibleError, Invertible, flip_coin


class ColorJitter(transforms.ColorJitter, Invertible):
    _transform = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        self._transform = self.get_params(self.brightness, self.contrast,
                                          self.saturation, self.hue)
        return self._transform(img)

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied.')

        return T.Lambda(
            lambd=lambda x: x,
            tf_inv=self._transform,
            repr_str='ColorJitterInverse()'
        )

    def _can_invert(self):
        return self._transform is not None


class Grayscale(transforms.Grayscale, Invertible):
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        return F.to_grayscale(img, num_output_channels=self.num_output_channels)

    def inverse(self):
        return T.Lambda(
            lambd=lambda x: x,
            tf_inv=Grayscale(self.num_output_channels),
            repr_str='GrayscaleInverse()'
        )


class RandomGrayscale(transforms.RandomGrayscale, Invertible):
    _transform = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
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
