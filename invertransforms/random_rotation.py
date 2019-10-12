from torchvision import transforms

from invertransforms import Lambda
from invertransforms import functional as F
from invertransforms.util import UndefinedInvertible


class RandomRotation(transforms.RandomRotation, UndefinedInvertible):
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

    def _invert(self, **kwargs):
        angle = self.angle
        return rotation_inverse(
            img_h=self.img_h,
            img_w=self.img_w,
            angle=angle,
            resample=self.resample,
            expand=self.expand,
            center=self.center,
        )

    def _can_invert(self):
        return self.angle is not None and self.img_w is not None and self.img_h is not None


def rotation_inverse(img_h, img_w, angle, resample, expand, center):
    return Lambda(
        lambd=lambda img: F.center_crop(
            img=F.rotate(img, -angle, resample, expand, center),
            output_size=(img_h, img_w)
        ),
        lambd_inv=lambda img: F.center_crop(
            img=F.rotate(img, angle, resample, expand, center),
            output_size=(img_h, img_w)
        ),
        repr_str=f'RotationInvert()'
    )
