Module invertransforms.sequence
===============================
Sequence Module

This module contains transformations that are applied to a list of transformations.
It can apply them in order, in random order, pick one randomly, or apply none, ...

Classes
-------

`Compose(transforms)`
:   Composes several transforms together.
    
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.Compose
    * invertransforms.util.invertible.Invertible

`RandomApply(transforms, p=0.5)`
:   Apply randomly a list of transformations with a given probability
    
    Args:
        transforms: one or multiple of transformations
        p (float): probability

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomApply
    * torchvision.transforms.transforms.RandomTransforms
    * invertransforms.util.invertible.Invertible

`RandomChoice(transforms)`
:   Apply single transformation randomly picked from a list

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomChoice
    * torchvision.transforms.transforms.RandomTransforms
    * invertransforms.util.invertible.Invertible

`RandomOrder(transforms)`
:   Apply a list of transformations in a random order

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomOrder
    * torchvision.transforms.transforms.RandomTransforms
    * invertransforms.util.invertible.Invertible