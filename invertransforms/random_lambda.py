from invertransforms.identity import Identity
from invertransforms.lambd import Lambda
from invertransforms.util import flip_coin


class RandomLambda(Lambda):
    def __init__(self, lambd, lambd_inv=None, p=0.5, repr_str=None):
        super().__init__(lambd=lambd, lambd_inv=lambd_inv, repr_str=repr_str)
        self.p = p
        self.__do_tf = None

    def __call__(self, img):
        self.__do_tf = flip_coin(self.p)
        if self.__do_tf:
            return self.lambd(img)
        return img

    def _invert(self):
        if self.__do_tf:
            return super()._invert()
        else:
            return Identity()

    def _can_invert(self):
        return self.__do_tf is not None

    def __repr__(self):
        if self.__repr_str is not None:
            return self.__repr_str
        return f'{self.__class__.__name__}(p={self.p})'
