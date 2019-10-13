import torch
from PIL import Image, PILLOW_VERSION
from torchvision import transforms
from torchvision.transforms.functional import _get_inverse_affine_matrix

from invertransforms import functional as F
from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleError


class Affine(Invertible):
    def __init__(self, matrix):
        self.matrix = matrix

    def invert(self):
        matrix_inv = _invert_affine_matrix(self.matrix)
        return Affine(matrix_inv)

    def __call__(self, img):
        return _affine_with_matrix(img, self.matrix)

    def __repr__(self):
        return f'{self.__class__.__name__}(matrix={self.matrix})'


class RandomAffine(transforms.RandomAffine, Invertible):
    _matrix = None

    def __call__(self, img):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        params = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
        self._matrix = _get_inverse_affine_matrix(center, *params)
        return F.affine(img, *params, resample=self.resample, fillcolor=self.fillcolor)

    def invert(self):
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
