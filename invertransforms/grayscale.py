from torchvision import transforms

import invertransforms as T
from invertransforms.util import Invertible, flip_coin, InvertibleException


class Grayscale(transforms.Grayscale, Invertible):
    def invert(self):
        return T.Lambda(
            lambd=lambda img: img,
            tf_inv=Grayscale(num_output_channels=self.num_output_channels),
            repr_str='GrayscaleInvert()'
        )


class RandomGrayscale(transforms.RandomGrayscale, Invertible):
    __tf = None

    def __call__(self, img):
        self.__tf = T.Identity()
        if flip_coin(self.p):
            num_output_channels = 1 if img.mode == 'L' else 3
            self.__tf = Grayscale(num_output_channels=num_output_channels)
        return self.__tf(img)

    def invert(self):
        if not self.__can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        return self.__tf.invert()

    def __can_invert(self):
        return self.__tf is not None
