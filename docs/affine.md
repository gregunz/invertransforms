Module invertransforms.affine
=============================
This module contains transform classes to apply affine transformations to images.
The transformation can be random or fixed.
Including specific transformations for rotations.

Classes
-------

`Affine(matrix)`
:   Apply affine transformation on the image.
    
    Args:
        matrix (list of int): transformation matrix (from destination image to source)
         because we want to interpolate the (discrete) destination pixel from the local
         area around the (floating) source pixel.

    ### Ancestors (in MRO)

    * invertransforms.lib.Invertible

`RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)`
:   Random affine transformation of the image keeping center invariant
    
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)
    
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomAffine
    * invertransforms.lib.Invertible

`RandomRotation(degrees, resample=False, expand=False, center=None)`
:   Rotate the image by angle.
    
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomRotation
    * invertransforms.lib.Invertible

`Rotation(angle, resample=False, expand=False, center=None)`
:   Rotate the image given an angle (in degrees).
    
    Args:
        angle (float or int): degrees of the angle
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.

    ### Ancestors (in MRO)

    * invertransforms.lib.Invertible