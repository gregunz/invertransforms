from torchvision import transforms

from invertransforms import functional as F
from invertransforms.lambd import Lambda
from invertransforms.util import UndefinedInvertible, flip_coin


class RandomPerspective(transforms.RandomPerspective, UndefinedInvertible):
    __startpoints = None
    __endpoints = None
    __do_tf = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be Perspectively transformed.

        Returns:
            PIL Image: Random perspectivley transformed image.
        """
        self.__do_tf = flip_coin(self.p)
        if self.__do_tf:
            width, height = img.size
            self.__startpoints, self.__endpoints = self.get_params(width, height, self.distortion_scale)
            return F.perspective(img, self.__startpoints, self.__endpoints, self.interpolation)
        return img

    def _invert(self):
        startpoints = self.__startpoints
        endpoints = self.__endpoints
        return Lambda(
            lambd=lambda img: F.perspective(img, endpoints, startpoints, self.interpolation),
            lambd_inv=lambda img: F.perspective(img, startpoints, endpoints, self.interpolation),
            repr_str=f'PerspectiveInvert()'
        )

    def _can_invert(self):
        return self.__do_tf is not None and self.__startpoints is not None and self.__endpoints is not None
