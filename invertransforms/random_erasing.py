from torchvision import transforms

import invertransforms as T
from invertransforms.util import Invertible


class RandomErasing(transforms.RandomErasing, Invertible):
    def inverse(self):
        return T.Identity()
