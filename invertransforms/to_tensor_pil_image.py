from torchvision import transforms

from invertransforms.util import Invertible


class ToPILImage(transforms.ToPILImage, Invertible):
    def invert(self):
        return ToTensor()


class ToTensor(transforms.ToTensor, Invertible):
    def invert(self):
        return ToPILImage()
