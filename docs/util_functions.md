Module invertransforms.util_functions
=====================================
This modules contains utility transformations for building a clean pipeline.

Classes
-------

`Identity(log_fn=<function Identity.<lambda>>)`
:   Returns its input image without changes.
    
    Args:
        log_fn (function): optional, function useful for logging/debugging.
    
    Returns its input.
    Output = Input
    
    Can be use for debugging/logging if a log_fn is provided.
    It is used throughout the library when inverse transformation is identity.

    ### Ancestors (in MRO)

    * invertransforms.lib.Invertible

`Lambda(lambd, tf_inv=None, repr_str=None)`
:   Apply a user-defined lambda as a transform.
    
    Args:
        lambd (function): Lambda/function to be used for transform
        tf_inv (function or Invertible): Invertible transform or Lambda/function to be returned by the `inverse` method
        repr_str (str): optional, overriding the output of __repr__.

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.Lambda
    * invertransforms.lib.Invertible

`ToPILImage(mode=None)`
:   Convert a tensor or an ndarray to PIL Image.
    
    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.
    
    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
             - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
             - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
             - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
             - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``,
               ``short``).
    
    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.ToPILImage
    * invertransforms.lib.Invertible

`ToTensor(*args, **kwargs)`
:   Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    
    In the other cases, tensors are returned without scaling.

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.ToTensor
    * invertransforms.lib.Invertible

`TransformIf(transform, condition)`
:   Apply a transformation if the condition is met.
    Otherwise, returns its input.
    
    Args:
          transform: a transformation
          condition (bool): a boolean

    ### Ancestors (in MRO)

    * invertransforms.lib.Invertible