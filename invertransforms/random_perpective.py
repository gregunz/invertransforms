from torchvision import transforms

from invertransforms.util import UndefinedInvertible


class RandomPerspective(transforms.RandomPerspective, UndefinedInvertible):
    raise NotImplementedError
