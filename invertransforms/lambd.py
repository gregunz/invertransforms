from torchvision import transforms

from invertransforms.util import Invertible
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
