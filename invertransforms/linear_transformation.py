from torchvision import transforms

from invertransforms.util.invertible import InvertibleException, Invertible


class LinearTransformation(transforms.LinearTransformation, Invertible):
    def _invert(self, **kwargs):
        try:
            return LinearTransformation(
                transformation_matrix=self.transformation_matrix.inverse(),
                mean_vector=(-1.0) * self.mean_vector @ self.transformation_matrix
            )
        except Exception as e:
            raise InvertibleException(
                f'{self.__repr__()} is not invertible because the transformation matrix is not invertible.')
