Module invertransforms.color
============================
This modules contains transformations on the image channels (RGB, grayscale).

Technically these transformations cannot be inverted or it simply makes not much sense,
hence the inverse is usually the identity function.

Classes
-------

`ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)`
:   Randomly change the brightness, contrast and saturation of an image.
    
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.ColorJitter
    * invertransforms.lib.Invertible

`Grayscale(num_output_channels=1)`
:   Convert image to grayscale.
    
    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image
    
    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.Grayscale
    * invertransforms.lib.Invertible

`RandomGrayscale(p=0.1)`
:   Randomly convert image to grayscale with a probability of p (default 0.1).
    
    Args:
        p (float): probability that image should be converted to grayscale.
    
    Returns:
        PIL Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomGrayscale
    * invertransforms.lib.Invertible