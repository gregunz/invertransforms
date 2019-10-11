from collections import Sequence

from torchvision import transforms
from torchvision.transforms import functional as F

from invertransforms.util import UndefinedInvertible


class Crop(UndefinedInvertible):
    img_w = None
    img_h = None

    def __init__(self, location, size):
        if isinstance(location, int):
            self.tl_i = self.tl_j = location
        elif isinstance(location, Sequence) and len(location) == 2:
            self.tl_i, self.tl_j = location
        else:
            raise Exception(f'Argument mismatch: location={location}')
        if isinstance(size, int):
            self.crop_h = self.crop_w = size
        elif isinstance(size, Sequence) and len(size) == 2:
            self.crop_h, self.crop_w = size
        else:
            raise Exception(f'Argument mismatch: size={size}')

    def __repr__(self):
        return f'{self.__class__.__name__}(location=({self.tl_i},{self.tl_j}), size=({self.crop_h},{self.crop_w}))'

    def __call__(self, img):
        self.img_w, self.img_h = img.size
        return F.crop(img, self.tl_i, self.tl_j, self.crop_h, self.crop_w)

    def _invert(self, **kwargs):
        padding = (
            self.tl_j,
            self.tl_i,
            self.img_w - self.crop_w - self.tl_j,
            self.img_h - self.crop_h - self.tl_i,
        )
        return Pad(padding=padding, **kwargs)

    def _can_invert(self):
        return self.img_w is not None and self.img_h is not None


class Pad(transforms.Pad, UndefinedInvertible):
    img_w = None
    img_h = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        self.img_w, self.img_h = img.size
        return super().__call__(img=img)

    def _invert(self):
        padding = self.padding
        if isinstance(padding, int):
            location = (padding, padding)
        elif isinstance(padding, Sequence) and len(padding) == 2:
            location = padding
        elif isinstance(padding, Sequence) and len(padding) == 4:
            location = padding[:2][::-1]
        else:
            raise Exception(f'Argument mismatch: padding={padding}')

        size = (self.img_h, self.img_w)
        print(location, size)
        return Crop(location=location, size=size)

    def _can_invert(self):
        return self.img_w is not None and self.img_h is not None
