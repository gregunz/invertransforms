import math
import random
import warnings

import invertransforms as T
from invertransforms.util import Invertible
from invertransforms.util.invertible import InvertibleException


class RandomRotCrop(Invertible):
    """
    Rotate and image and extract a crop withing the rotated region.
    """

    def __init__(self, degrees=(0, 360), crop_size=None):
        if isinstance(degrees, int) or isinstance(degrees, float):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degree_low, self.degree_high = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degree_low, self.degree_high = degrees

        if crop_size is None:
            self.out_h, self.out_w = None, None
            warnings.warn('only square image are handled properly, doing a center crop of max size')
        elif isinstance(crop_size, int):
            self.out_h, self.out_w = (crop_size, crop_size)
        else:
            if len(degrees) != 2:
                raise ValueError("If crop size is a sequence, it must be of len 2.")
            self.out_h, self.out_w = crop_size
            if not isinstance(self.out_w, int) or not isinstance(self.out_h, int):
                raise ValueError(f'crop size must be an integer, found {crop_size}')
            if self.out_h != self.out_w:
                # todo: solve this
                raise ValueError('only handling square crop for now')

        self.transforms = []

    def __call__(self, img):
        self.transforms = []

        w, h = img.size  # PIL image
        if w != h:
            # todo: should handle rectangle image as well
            warnings.warn('only square image are handled properly, doing a center crop of max size')
            center_crop = T.CenterCrop(size=min(w, h))
            img = center_crop(img)
            self.transforms.append(center_crop)
            w, h = img.size

        angle = random.uniform(self.degree_low, self.degree_high)
        out_w_max, out_h_max = _rotated_rect_with_max_area(w=w, h=h, angle=angle)  # max area rectangle of rotated image

        out_h = self.out_h
        out_w = self.out_w

        if out_h is None and out_w is None:
            out_h = out_h_max
            out_w = out_w_max

        if out_h > out_h_max or out_w > out_w_max:
            warnings.warn('cropping size is bigger than maximum size crop within the rotated area')
            out_h = out_h_max
            out_w = out_w_max

        out_h_init = round(h / out_h_max * out_h)
        out_w_init = round(w / out_w_max * out_w)

        h_margin = (h - out_h_init) // 2
        w_margin = (w - out_w_init) // 2

        i_img_center = h / 2
        j_img_center = w / 2

        i_crop_center = i_img_center + random.randint(-h_margin, h_margin)
        j_crop_center = j_img_center + random.randint(-w_margin, w_margin)

        i_crop_center_rot, j_crop_center_rot = \
            _rotate_coordinates((i_crop_center, j_crop_center), angle, (i_img_center, j_img_center))

        rotate = T.Rotation(angle, expand=True)
        img = rotate(img)
        self.transforms.append(rotate)
        w_rot, h_rot = img.size

        i = round(i_crop_center_rot + (h_rot - h - out_h) / 2)
        j = round(j_crop_center_rot + (w_rot - h - out_w) / 2)

        crop = T.Crop(location=(j, i), size=(int(out_h), int(out_w)))
        img = crop(img)
        self.transforms.append(crop)

        if (self.out_w is not None and self.out_w != out_w) or \
                (self.out_h is not None and self.out_h != out_h):
            center_crop = T.CenterCrop(size=(self.out_h, self.out_w))
            img = center_crop(img)
            self.transforms.append(center_crop)

        return img

    def __repr__(self):
        return f'RandomRotCrop(' \
               f'degrees=[{self.degree_low}, {self.degree_high}], ' \
               f'crop_size=({self.out_h}, {self.out_w})' \
               f')'

    def invert(self):
        if not self._can_invert():
            raise InvertibleException('Cannot invert a random transformation before it is applied.')

        return T.Compose(self.transforms).invert()

    def _can_invert(self):
        return len(self.transforms) > 0


def _rotate_coordinates(coordinates, angle, center_coordinates):
    x, y = coordinates
    center_x, center_y = center_coordinates
    angle = angle * math.pi / 180

    sin_a, cos_a = math.sin(angle), math.cos(angle)

    x -= center_x
    y -= center_y
    x_rot = cos_a * x + sin_a * y
    y_rot = cos_a * y - sin_a * x
    return x_rot + center_x, y_rot + center_y


def _rotated_rect_with_max_area(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    degrees), computes the width and height of the largest possible
    (maximal area) axis-aligned rectangle within the rotated rectangle.

    :param w: int, width
    :param h: int, height
    :param angle: float, angle in degrees
    :return: tuple(float, float), width and height
    """
    angle = angle * math.pi / 180
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # it suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        # the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr
