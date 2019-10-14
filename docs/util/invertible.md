Module invertransforms.util.invertible
======================================
Invertible Module

This module contains the basic building block to make transformations invertible.

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
    * invertransforms.sequence.RandomApply
    * invertransforms.sequence.RandomChoice
    * invertransforms.sequence.RandomOrder
    * invertransforms.sequence.Compose
    * invertransforms.resize.Resize
    * invertransforms.resize.RandomResizedCrop
    * invertransforms.tensors.LinearTransformation
    * invertransforms.tensors.Normalize
    * invertransforms.tensors.RandomErasing
    * invertransforms.util_functions.Identity
    * invertransforms.util_functions.Lambda
    * invertransforms.util_functions.TransformIf
    * invertransforms.util_functions.ToPILImage
    * invertransforms.util_functions.ToTensor

    ### Methods

    `apply(self, img)`
    :   Apply the transformation.
        This is an alias to the `__call__` method which should be preferred.
        Its main purpose is to appear in the doc alongside `inverse` and `replay`.
        
        Args:
            img (PIL Image, torch.Tensor, Any): input image
        
        Returns: image

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

`InvertibleError(message)`
:   Error raised when transformation cannot be inverted.

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException