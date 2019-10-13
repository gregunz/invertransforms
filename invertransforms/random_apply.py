from torchvision import transforms

from invertransforms.compose import Compose
from invertransforms.identity import Identity
from invertransforms.util import Invertible, flip_coin
from invertransforms.util.invertible import InvertibleException


class RandomApply(transforms.RandomApply, Invertible):
    __do_tf = None

    def __call__(self, img):
        self.__do_tf = flip_coin(self.p)
        if self.__do_tf:
            for t in self.transforms:
                img = t(img)
        return img

    def invert(self):
        if not self.__can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        if self.__do_tf:
            return Compose(self.transforms).invert()
        else:
            return Identity()

    def __can_invert(self):
        return self.__do_tf is not None
