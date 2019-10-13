from torchvision import transforms

from invertransforms.util.invertible import InvertibleException, Invertible


class LinearTransformation(transforms.LinearTransformation, Invertible):
    def invert(self):
        try:
            return LinearTransformation(
                transformation_matrix=self.transformation_matrix.inverse(),
                mean_vector=(-1.0) * self.mean_vector @ self.transformation_matrix
            )
        except RuntimeError:
            raise InvertibleException(
                f'{self.__repr__()} is not invertible because the transformation matrix singular.')
