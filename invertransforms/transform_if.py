from invertransforms import Identity
from invertransforms.util import Invertible, _Invertible


class TransformIf(Invertible):
    def __init__(self, transform, condition: bool):
        if condition:
            assert isinstance(transform, _Invertible), f'{transform.__class__.__name__} is not an invertible class'
            self.transform = transform
        else:
            self.transform = Identity()

    def __call__(self, img):
        return self.transform.__call__(img)

    def __repr__(self):
        return self.transform.__repr__()

    def _invert(self, **kwargs):
        return self.transform._invert(**kwargs)
