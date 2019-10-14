"""
Affine Module.

This module contains transform classes to apply affine transformations to images.
The transformation can be random or fixed.
Including specific transformations for rotations.

"""
import torch
from PIL import Image, PILLOW_VERSION
from torchvision import transforms
from torchvision.transforms.functional import _get_inverse_affine_matrix

from invertransforms import functional as F
from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleError


class Affine(Invertible):
    """
    Apply affine transformation on the image.

    Args:
        matrix (list of int): transformation matrix (from destination image to source)
         because we want to interpolate the (discrete) destination pixel from the local
         area around the (floating) source pixel.
    """

    def __init__(self, matrix):
        self.matrix = matrix

    def inverse(self):
        matrix_inv = _invert_affine_matrix(self.matrix)
        return Affine(matrix_inv)

    def __call__(self, img):
        """
        Args
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        return _affine_with_matrix(img, self.matrix)

    def __repr__(self):
        return f'{self.__class__.__name__}(matrix={self.matrix})'


class RandomAffine(transforms.RandomAffine, Invertible):
    _matrix = None

    def __call__(self, img):
        """
        Args
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        params = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
        self._matrix = _get_inverse_affine_matrix(center, *params)
        return F.affine(img, *params, resample=self.resample, fillcolor=self.fillcolor)

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied.')

        matrix_inv = _invert_affine_matrix(self._matrix)
        return Affine(matrix_inv)

    def _can_invert(self):
        return self._matrix is not None


def _affine_with_matrix(img, matrix, resample=0, fillcolor=None):
    kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] >= '5' else {}
    return img.transform(img.size, Image.AFFINE, matrix, resample, **kwargs)


def _invert_affine_matrix(matrix):
    if len(matrix) == 6:
        matrix += [0., 0., 1.]
    matrix = torch.tensor(matrix).reshape(3, 3)
    return matrix.inverse().reshape(-1).tolist()


class Rotation(Invertible):
    def __init__(self, angle, resample=False, expand=False, center=None):
        self.angle = angle
        self.resample = resample
        self.expand = expand
        self.center = center
        self._img_h = self._img_w = None

    def __call__(self, img):
        first_call = self._img_h is None or self._img_w is None
        if first_call:
            self._img_w, self._img_h = img.size
        img = F.rotate(img, -self.angle, self.resample, self.expand, self.center)
        if not first_call and self.expand:
            img = F.center_crop(img=img, output_size=(self._img_h, self._img_w))
        return img

    def inverse(self):
        if (self._img_h is None or self._img_w is None) and self.expand:
            raise InvertibleError(
                'Cannot invert a transformation before it is applied'
                ' (size of image before expanded rotation unknown).')  # note: the size could be computed
        rot = Rotation(
            angle=-self.angle,
            resample=self.resample,
            expand=self.expand,
            center=self.center,
        )
        rot._img_h, rot._img_w = self._img_h, self._img_w
        return rot

    def __repr__(self):
        format_string = self.__class__.__name__ + '(angle={0}'.format(self.angle)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomRotation(transforms.RandomRotation, Invertible):
    _angle = _img_h = _img_w = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        self._angle = self.get_params(self.degrees)
        self._img_w, self._img_h = img.size
        return F.rotate(img, self._angle, self.resample, self.expand, self.center)

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied.')

        rot = Rotation(
            angle=-self._angle,
            resample=self.resample,
            expand=self.expand,
            center=self.center,
        )
        rot._img_h, rot._img_w = self._img_h, self._img_w
        return rot

    def _can_invert(self):
        return self._angle is not None and self._img_w is not None and self._img_h is not None
