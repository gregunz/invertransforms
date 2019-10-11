from torchvision import transforms

from invertransforms import Identity, Compose
from invertransforms.util import UndefinedInvertible, flip_coin


class RandomApply(transforms.RandomApply, UndefinedInvertible):
    do_tf = None

    def __call__(self, img):
        self.do_tf = flip_coin(self.p)
        if self.do_tf:
            for t in self.transforms:
                img = t(img)
        return img

    def _invert(self):
        if self.do_tf:
            return Compose(self.transforms).invert()
        else:
            return Identity()

    def _can_invert(self):
        return self.do_tf is not None
