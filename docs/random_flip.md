Module invertransforms.random_flip
==================================

Classes
-------

`HorizontalFlip(*args, **kwargs)`
:   

    ### Ancestors (in MRO)

    * invertransforms.util.invertible.Invertible

`RandomHorizontalFlip(p=0.5)`
:   Horizontally flip the given PIL Image randomly with a given probability.
    
    Args:
        p (float): probability of the image being flipped. Default value is 0.5

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomHorizontalFlip
    * invertransforms.util.invertible.Invertible

`RandomVerticalFlip(p=0.5)`
:   Vertically flip the given PIL Image randomly with a given probability.
    
    Args:
        p (float): probability of the image being flipped. Default value is 0.5

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomVerticalFlip
    * invertransforms.util.invertible.Invertible

`VerticalFlip(*args, **kwargs)`
:   

    ### Ancestors (in MRO)

    * invertransforms.util.invertible.Invertible