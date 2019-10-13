from torchvision import transforms

import invertransforms as T
from invertransforms.util import Invertible, flip_coin
from invertransforms.util.invertible import InvertibleException


class Lambda(transforms.Lambda, Invertible):
    def __init__(self, lambd, tf_inv=None, repr_str=None):
        super().__init__(lambd=lambd)
        assert repr_str is None or isinstance(repr_str, str), 'Expecting a string for repr_str argument'
        self._repr_str = repr_str
        assert tf_inv is None or callable(tf_inv), repr(type(tf_inv).__name__) + " object is not callable"
        self.tf_inv = tf_inv

    def invert(self):
        if self.tf_inv is None:
            raise InvertibleException('Cannot invert transformation, tf_inv_builder is None')
        if isinstance(self.tf_inv, Invertible):
            return self.tf_inv
        else:
            repr_str = repr(self)
            suffix = 'Invert()'
            if suffix in repr_str:
                repr_str = repr_str[:-len(suffix)]
            else:
                repr_str += suffix
            return Lambda(
                lambd=self.tf_inv,
                tf_inv=self.lambd,
                repr_str=repr_str,
            )

    def __repr__(self):
        if self._repr_str is not None:
            return self._repr_str
        return super().__repr__()


class RandomLambda(Lambda):
    def __init__(self, lambd, tf_inv=None, p=0.5, repr_str=None):
        super().__init__(lambd=lambd, tf_inv=tf_inv, repr_str=repr_str)
        self.p = p
        self.tf = None

    def __call__(self, img):
        self.tf = T.Identity()
        if flip_coin(self.p):
            self.tf = super()
        return self.tf.__call__(img)  # use of call because super is not callable

    def __repr__(self):
        if self._repr_str is not None:
            return self._repr_str
        return f'{self.tf.__repr__()}(p={self.p})'

    def invert(self):
        if not self.__can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        return self.tf.invert()

    def __can_invert(self):
        return self.tf is not None
