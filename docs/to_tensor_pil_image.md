Module invertransforms.to_tensor_pil_image
==========================================

Classes
-------

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
    * invertransforms.util.invertible.Invertible

`ToTensor(*args, **kwargs)`
:   Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    
    In the other cases, tensors are returned without scaling.

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.ToTensor
    * invertransforms.util.invertible.Invertible