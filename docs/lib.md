Module invertransforms.lib
==========================
This module contains the basic building blocks of this library.
It contains the abstract class all transformations should extend
and utility functions.

Functions
---------

    
`flip_coin(p)`
:   Return true with probability p
    
    Args:
        p: float, probability to return True
    
    Returns: bool

Classes
-------

`Invertible(*args, **kwargs)`
:   

    ### Descendants

    * invertransforms.affine.Affine
    * invertransforms.affine.RandomAffine
    * invertransforms.affine.Rotation
    * invertransforms.affine.RandomRotation
    * invertransforms.color.ColorJitter
    * invertransforms.color.Grayscale
    * invertransforms.color.RandomGrayscale
    * invertransforms.crop_pad.Crop
    * invertransforms.crop_pad.CenterCrop
    * invertransforms.crop_pad.RandomCrop
    * invertransforms.crop_pad.Pad
    * invertransforms.crop_pad.FiveCrop
    * invertransforms.crop_pad.TenCrop
    * invertransforms.perpective.Perspective
    * invertransforms.perpective.RandomPerspective
    * invertransforms.perpective.HorizontalFlip
    * invertransforms.perpective.RandomHorizontalFlip
    * invertransforms.perpective.VerticalFlip
    * invertransforms.perpective.RandomVerticalFlip
    * invertransforms.resize.Resize
    * invertransforms.resize.RandomResizedCrop
    * invertransforms.sequence.RandomApply
    * invertransforms.sequence.RandomChoice
    * invertransforms.sequence.RandomOrder
    * invertransforms.sequence.Compose
    * invertransforms.tensors.LinearTransformation
    * invertransforms.tensors.Normalize
    * invertransforms.tensors.RandomErasing
    * invertransforms.util_functions.Identity
    * invertransforms.util_functions.Lambda
    * invertransforms.util_functions.TransformIf
    * invertransforms.util_functions.ToPILImage
    * invertransforms.util_functions.ToTensor

    ### Class variables

    `tracked_inverses`
    :   dict() -> new empty dictionary
        dict(mapping) -> new dictionary initialized from a mapping object's
            (key, value) pairs
        dict(iterable) -> new dictionary initialized as if via:
            d = {}
            for k, v in iterable:
                d[k] = v
        dict(**kwargs) -> new dictionary initialized with the name=value pairs
            in the keyword argument list.  For example:  dict(one=1, two=2)

    ### Methods

    `get_inverse(self, index)`
    :   Get the inverse of a tracked transformation given its index.
        
        Args:
            index (int or Any): index associated with the tracked inverse transform
        
        Returns:
            inverse transformation

    `inverse(self)`
    :   Abstract method to return the inverse of the transformation
        
        Returns (Invertible): tf

    `invert(self, img)`
    :   Apply the inverse of this transformation.
        
        Args:
            img (PIL Image, torch.Tensor, Any): input image
        
        Returns: image

    `replay(self, img)`
    :   Replay a transformation (with random like previous runs).
        If it is called before any calls to `__call__`, it will simply calls `__call__`
        
        Note: Any call to `__call__` will change the randomness again.
        
        Args:
            img (PIL Image, torch.Tensor, Any): input image
        
        Returns: image

    `track(self, img, index=None)`
    :   Apply the transformation and track all inverses.
        
        Args:
            img (PIL Image, torch.Tensor, Any): input image.
            index (optional, int or Any): index associated with the tracked inverse transform; increasing int when not defined
        
        Returns: image

`InvertibleError(message)`
:   Error raised when transformation cannot be inverted.

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException