from torchvision import transforms

from invertransforms import functional as F
from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleException


class Rotation(Invertible):
    def __init__(self, angle, resample=False, expand=False, center=None):
        self.angle = angle
        self.resample = resample
        self.expand = expand
        self.center = center
        self._img_h = self._img_w = None

    def __call__(self, img):
        first_call = self._img_h is None or self._img_w is None
        if first_call:
            self._img_w, self._img_h = img.size
        img = F.rotate(img, -self.angle, self.resample, self.expand, self.center)
        if not first_call and self.expand:
            img = F.center_crop(img=img, output_size=(self._img_h, self._img_w))
        return img

    def invert(self):
        if (self._img_h is None or self._img_w is None) and self.expand:
            raise InvertibleException(
                'Cannot invert a transformation before it is applied'
                ' (size of image before expanded rotation unknown).')  # note: the size could be computed
        rot = Rotation(
            angle=-self.angle,
            resample=self.resample,
            expand=self.expand,
            center=self.center,
        )
        rot._img_h, rot._img_w = self._img_h, self._img_w
        return rot

    def __repr__(self):
        format_string = self.__class__.__name__ + '(angle={0}'.format(self.angle)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomRotation(transforms.RandomRotation, Invertible):
    angle = img_h = img_w = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        self.angle = self.get_params(self.degrees)
        self.img_w, self.img_h = img.size
        return F.rotate(img, self.angle, self.resample, self.expand, self.center)

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        rot = Rotation(
            angle=-self.angle,
            resample=self.resample,
            expand=self.expand,
            center=self.center,
        )
        rot._img_h, rot._img_w = self.img_h, self.img_w
        return rot

    def _can_invert(self):
        return self.angle is not None and self.img_w is not None and self.img_h is not None
