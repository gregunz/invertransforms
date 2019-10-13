Module invertransforms.resize
=============================

Classes
-------

`Resize(size, interpolation=2)`
:   Resize the input PIL Image to the given size.
    
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.Resize
    * invertransforms.util.invertible.Invertible

    ### Descendants

    * invertransforms.resize.Scale

`Scale(*args, **kwargs)`
:   Note: This transform is deprecated in favor of Resize.

    ### Ancestors (in MRO)

    * invertransforms.resize.Resize
    * torchvision.transforms.transforms.Resize
    * invertransforms.util.invertible.Invertible