Module invertransforms.perpective
=================================
This module contains transformations for perspective transformation and flipping vertically or horizontally images.
These transformations can be applied deterministically or randomly.

Classes
-------

`HorizontalFlip(*args, **kwargs)`
:   Flip the image horizontally.

    ### Ancestors (in MRO)

    * invertransforms.lib.Invertible

`Perspective(startpoints, endpoints, interpolation=3)`
:   

    ### Ancestors (in MRO)

    * invertransforms.lib.Invertible

`RandomHorizontalFlip(p=0.5)`
:   Horizontally flip the given PIL Image randomly with a given probability.
    
    Args:
        p (float): probability of the image being flipped. Default value is 0.5

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomHorizontalFlip
    * invertransforms.lib.Invertible

`RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)`
:   Performs Perspective transformation of the given PIL Image randomly with a given probability.
    
    Args:
        interpolation : Default- Image.BICUBIC
    
        p (float): probability of the image being perspectively transformed. Default value is 0.5
    
        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomPerspective
    * invertransforms.lib.Invertible

`RandomVerticalFlip(p=0.5)`
:   Vertically flip the given PIL Image randomly with a given probability.
    
    Args:
        p (float): probability of the image being flipped. Default value is 0.5

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomVerticalFlip
    * invertransforms.lib.Invertible

`VerticalFlip(*args, **kwargs)`
:   Flip the image vertically.

    ### Ancestors (in MRO)

    * invertransforms.lib.Invertible