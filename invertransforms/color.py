"""
Color Module

This modules contains transformations on the image channels (RBG, grayscale).
Generally this transformation cannot be reversed or it simply makes not much sense.
"""
from torchvision import transforms
from torchvision.transforms import functional as F

import invertransforms as T
from invertransforms.util import Invertible, flip_coin, InvertibleError


class ColorJitterFixed(Invertible):
    """
    Change the brightness, contrast and saturation of an image.
    Not random version for `ColorJitter`.
    
    Args:
        brightness (float): How much to jitter brightness.
        contrast (float): How much to jitter contrast.
        saturation (float): How much to jitter saturation.
        hue (float): How much to jitter hue.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.transform = T.Compose([
            T.Lambda(lambda img: F.adjust_brightness(img, self.brightness)),
            T.Lambda(lambda img: F.adjust_contrast(img, self.contrast)),
            T.Lambda(lambda img: F.adjust_saturation(img, self.saturation)),
            T.Lambda(lambda img: F.adjust_hue(img, self.hue)),
        ])

    def __call__(self, img):
        return self.transform(img)

    def inverse(self):
        return T.Lambda(
            lambd=lambda x: x,
            tf_inv=ColorJitterFixed(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            ),
            repr_str='ColorJitterInverse()'
        )


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


class Grayscale(transforms.Grayscale, Invertible):
    def __call__(self, img):
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
