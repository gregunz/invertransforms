"""
This module contains transformations on torch tensors.
"""
import torch
from torchvision import transforms

import invertransforms as T
from invertransforms import functional as F
from invertransforms.lib import InvertibleError, Invertible, flip_coin


class LinearTransformation(transforms.LinearTransformation, Invertible):
    def inverse(self):
        try:
            return LinearTransformation(
                transformation_matrix=self.transformation_matrix.inverse(),
                mean_vector=(-1.0) * self.mean_vector @ self.transformation_matrix
            )
        except RuntimeError:
            raise InvertibleError(
                f'{self.__repr__()} is not invertible because the transformation matrix singular.')


class Normalize(transforms.Normalize, Invertible):
    def inverse(self):
        mean = torch.as_tensor(self.mean)
        std = torch.as_tensor(self.std)
        std_inv = torch.tensor(1.0) / std
        mean_inv = (-1.0) * mean * std_inv
        return Normalize(mean=mean_inv, std=std_inv, inplace=self.inplace)


class RandomErasing(transforms.RandomErasing, Invertible):
    _transform = None

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        self._transform = T.Identity()
        if flip_coin(self.p):
            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)
            self._transform = T.Lambda(
                lambd=lambda img: F.erase(img, x, y, h, w, v, self.inplace),
                tf_inv=lambda img: img,
                repr_str='RandomErasing()'
            )
        return img

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied.')
        return self._transform.inverse()

    def _can_invert(self):
        return self._transform is not None
