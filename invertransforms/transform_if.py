from invertransforms.identity import Identity
from invertransforms.util import Invertible, InvertibleException


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

    def invert(self):
        if not isinstance(self.transform, Invertible):
            raise InvertibleException(
                f'{self.transform} ({self.transform.__class__.__name__}) is not an invertible object')
        return self.transform.invert()
