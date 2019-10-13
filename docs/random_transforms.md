Module invertransforms.random_transforms
========================================

Classes
-------

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