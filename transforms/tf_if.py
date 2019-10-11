from transforms.util import Invertible


class TransformIf(Invertible):
    def __init__(self, condition: bool, transform):
        self.condition = condition
        self.transform = transform

    def __call__(self, inputs):
        if self.condition:
            return self.transform(inputs)
        return inputs

    def __repr__(self):
        if self.condition:
            return self.transform.__repr__()
        return 'Lambda x: x'

    def _invert(self, **kwargs):
        tf_inv = self.transform.invert(**kwargs) if self.condition else None
        return TransformIf(condition=self.condition, transform=tf_inv)
