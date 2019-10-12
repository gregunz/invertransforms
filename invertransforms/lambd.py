from torchvision import transforms

from invertransforms.util import Invertible


class Lambda(transforms.Lambda, Invertible):
    def __init__(self, lambd, lambd_inv=None, repr_str=None):
        super().__init__(lambd)
        assert repr_str is None or isinstance(repr_str, str), 'Expecting a string for repr_str argument'
        self.__repr_str = repr_str
        if lambd_inv is None:
            lambd_inv = lambd  # todo: what's best ? -no default, -lambda x:x, lambd_inv=lambd
        assert callable(lambd_inv), repr(type(lambd_inv).__name__) + " object is not callable"
        self.lambd_inv = lambd_inv

    def _invert(self):
        return Lambda(
            lambd=self.lambd_inv,
            lambd_inv=self.lambd,
            repr_str=self.__repr_str,
        )

    def __repr__(self):
        if self.__repr_str is not None:
            return self.__repr_str
        return super().__repr__()
