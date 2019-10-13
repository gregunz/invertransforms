Module invertransforms.center_crop
==================================

Classes
-------

`CenterCrop(size)`
:   Crops the given PIL Image at the center.
    
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.CenterCrop
    * invertransforms.util.invertible.Invertible