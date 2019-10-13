Module invertransforms.util
===========================

Sub-modules
-----------
* invertransforms.util.invertible

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
    * invertransforms.center_crop.CenterCrop
    * invertransforms.functions.Identity
    * invertransforms.functions.Lambda
    * invertransforms.functions.TransformIf
    * invertransforms.color_jitter.ColorJitter
    * invertransforms.compose.Compose
    * invertransforms.crop.Crop
    * invertransforms.crop.RandomCrop
    * invertransforms.five_crop.FiveCrop
    * invertransforms.grayscale.Grayscale
    * invertransforms.grayscale.RandomGrayscale
    * invertransforms.linear_transformation.LinearTransformation
    * invertransforms.normalize.Normalize
    * invertransforms.pad.Pad
    * invertransforms.perpective.Perspective
    * invertransforms.perpective.RandomPerspective
    * invertransforms.random_erasing.RandomErasing
    * invertransforms.random_flip.HorizontalFlip
    * invertransforms.random_flip.RandomHorizontalFlip
    * invertransforms.random_flip.VerticalFlip
    * invertransforms.random_flip.RandomVerticalFlip
    * invertransforms.random_resized_crop.RandomResizedCrop
    * invertransforms.random_transforms.RandomApply
    * invertransforms.random_transforms.RandomChoice
    * invertransforms.random_transforms.RandomOrder
    * invertransforms.resize.Resize
    * invertransforms.rotation.Rotation
    * invertransforms.rotation.RandomRotation
    * invertransforms.ten_crop.TenCrop
    * invertransforms.to_tensor_pil_image.ToPILImage
    * invertransforms.to_tensor_pil_image.ToTensor

    ### Methods

    `apply(self, img)`
    :   Apply the transformation.
        This is an alias to the `__call__` method which should be preferred.
        Its main purpose is to appear in the doc alongside `inverse` and `replay`.
        
        Args:
            img (PIL Image, torch.Tensor, Any): input image
        
        Returns: image

    `inverse(self, img)`
    :   Apply the inverse of this transformation.
        
        Args:
            img (PIL Image, torch.Tensor, Any): input image
        
        Returns: image

    `invert(self)`
    :   Abstract method to return the inverse of the transformation
        
        Returns (Invertible): tf

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