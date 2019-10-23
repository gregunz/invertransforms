Module invertransforms.resize
=============================
This modules contains transformations which resize images.

Classes
-------

`RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)`
:   Crop the given PIL Image to random size and aspect ratio.
    
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomResizedCrop
    * invertransforms.lib.Invertible

    ### Descendants

    * invertransforms.resize.RandomSizedCrop

`RandomSizedCrop(*args, **kwargs)`
:   Note: This transform is deprecated in favor of RandomResizedCrop.

    ### Ancestors (in MRO)

    * invertransforms.resize.RandomResizedCrop
    * torchvision.transforms.transforms.RandomResizedCrop
    * invertransforms.lib.Invertible

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
    * invertransforms.lib.Invertible

    ### Descendants

    * invertransforms.resize.Scale

`Scale(*args, **kwargs)`
:   Note: This transform is deprecated in favor of Resize.

    ### Ancestors (in MRO)

    * invertransforms.resize.Resize
    * torchvision.transforms.transforms.Resize
    * invertransforms.lib.Invertible