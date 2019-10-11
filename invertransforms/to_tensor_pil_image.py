from torchvision import transforms

from invertransforms.util import Invertible


class ToPILImage(transforms.ToPILImage, Invertible):
    def _invert(self):
        return ToTensor()


class ToTensor(transforms.ToTensor, Invertible):
    def _invert(self):
        return ToPILImage()
