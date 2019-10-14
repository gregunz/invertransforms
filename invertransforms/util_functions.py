"""
Util Functions Module

This modules contains utility transformations for building a clean pipeline.

"""
from torchvision import transforms

from invertransforms.util import Invertible, InvertibleError


class Identity(Invertible):
    def __call__(self, img):
        return img

    def inverse(self):
        return Identity()


class Lambda(transforms.Lambda, Invertible):
    def __init__(self, lambd, tf_inv=None, repr_str=None):
        super().__init__(lambd=lambd)
        assert repr_str is None or isinstance(repr_str, str), 'Expecting a string for repr_str argument'
        self._repr_str = repr_str
        assert tf_inv is None or callable(tf_inv), repr(type(tf_inv).__name__) + " object is not callable"
        self.tf_inv = tf_inv

    def inverse(self):
        if self.tf_inv is None:
            raise InvertibleError('Cannot invert transformation, tf_inv_builder is None')
        if isinstance(self.tf_inv, Invertible):
            return self.tf_inv
        else:
            repr_str = repr(self).split('()')[0]
            suffix = 'Inverse'
            if suffix in repr_str:
                repr_str = repr_str[:-len(suffix)]
            else:
                repr_str += suffix + '()'
            return Lambda(
                lambd=self.tf_inv,
                tf_inv=self.lambd,
                repr_str=repr_str,
            )

    def __repr__(self):
        if self._repr_str is not None:
            return self._repr_str
        return super().__repr__()


class TransformIf(Invertible):
    def __init__(self, transform, condition: bool):
        if condition:
            self.transform = transform
        else:
            self.transform = Identity()

    def __call__(self, img):
        return self.transform.__call__(img)

    def __repr__(self):
        return self.transform.__repr__()

    def inverse(self):
        if not isinstance(self.transform, Invertible):
            raise InvertibleError(
                f'{self.transform} ({self.transform.__class__.__name__}) is not an invertible object')
        return self.transform.inverse()


class ToPILImage(transforms.ToPILImage, Invertible):
    def inverse(self):
        return ToTensor()


class ToTensor(transforms.ToTensor, Invertible):
    def inverse(self):
        return ToPILImage()
