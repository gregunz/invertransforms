import torch
from PIL import Image, PILLOW_VERSION
from torchvision import transforms
from torchvision.transforms.functional import _get_inverse_affine_matrix

from invertransforms import Lambda
from invertransforms.util import UndefinedInvertible


class RandomAffine(transforms.RandomAffine, UndefinedInvertible):
    matrix = None

    def get_params(self, degrees, translate, scale_ranges, shears, img_size):
        params = super().get_params(degrees, translate, scale_ranges, shears, img_size)
        center = (img_size[0] * 0.5 + 0.5, img_size[1] * 0.5 + 0.5)
        angle, translate, scale, shear = params
        self.matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
        return params

    def _invert(self, **kwargs):
        matrix = self.matrix
        matrix_inv = invert_affine_matrix(matrix)
        return Lambda(
            lambd=lambda img: affine_matrix(img, matrix_inv, resample=self.resample, fillcolor=self.fillcolor),
            lambd_inv=lambda img: affine_matrix(img, matrix, resample=self.resample, fillcolor=self.fillcolor),
            repr_str='AffineInvert()'
        )

    def _can_invert(self):
        return self.matrix is None


def affine_matrix(img, matrix, resample=0, fillcolor=None):
    kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] >= '5' else {}
    return img.transform(img.size, Image.AFFINE, matrix, resample, **kwargs)


def invert_affine_matrix(matrix):
    if len(matrix) == 6:
        matrix += [0., 0., 1.]
    matrix = torch.tensor(matrix).reshape(3, 3)
    return matrix.inverse().reshape(-1).tolist()
