Module invertransforms.grayscale
================================

Classes
-------

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
    * invertransforms.util.invertible.Invertible

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
    * invertransforms.util.invertible.Invertible