from invertransforms.util import Invertible


class Identity(Invertible):
    def __call__(self, img):
        return img

    def invert(self):
        return Identity()
