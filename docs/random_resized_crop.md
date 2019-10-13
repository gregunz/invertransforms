Module invertransforms.random_resized_crop
==========================================

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
    * invertransforms.util.invertible.Invertible

    ### Descendants

    * invertransforms.random_resized_crop.RandomSizedCrop

`RandomSizedCrop(*args, **kwargs)`
:   Note: This transform is deprecated in favor of RandomResizedCrop.

    ### Ancestors (in MRO)

    * invertransforms.random_resized_crop.RandomResizedCrop
    * torchvision.transforms.transforms.RandomResizedCrop
    * invertransforms.util.invertible.Invertible