import random

from torchvision import transforms

from invertransforms.util import UndefinedInvertible


class RandomChoice(transforms.RandomOrder, UndefinedInvertible):
    transform = None

    def __call__(self, img):
        i = random.randint(0, len(self.transforms) - 1)
        self.transform = self.transforms[i]
        return self.transform(img)

    def _invert(self):
        return self.transform.invert()

    def _can_invert(self):
        return self.transform is not None
