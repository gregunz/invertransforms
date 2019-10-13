Module invertransforms.compose
==============================

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