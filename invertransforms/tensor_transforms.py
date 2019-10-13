import torch
from torchvision import transforms

from invertransforms.util.invertible import InvertibleError, Invertible


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
