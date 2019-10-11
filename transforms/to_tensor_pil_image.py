from torchvision import transforms

from transforms.util import Invertible


class ToPILImage(transforms.ToPILImage, Invertible):
    def _invert(self):
        return ToTensor()


class ToTensor(transforms.ToTensor, Invertible):
    def _invert(self):
        return ToPILImage()
