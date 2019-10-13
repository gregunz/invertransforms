Module invertransforms.rotation
===============================

Classes
-------

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
    * invertransforms.util.invertible.Invertible

`Rotation(angle, resample=False, expand=False, center=None)`
:   

    ### Ancestors (in MRO)

    * invertransforms.util.invertible.Invertible