from torchvision import transforms

from invertransforms.util import Invertible


class ToPILImage(transforms.ToPILImage, Invertible):
    def inverse(self):
        return ToTensor()


class ToTensor(transforms.ToTensor, Invertible):
    def inverse(self):
        return ToPILImage()
