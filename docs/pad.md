Module invertransforms.pad
==========================

Classes
-------

`Pad(padding, fill=0, padding_mode='constant')`
:   Pad the given PIL Image on all sides with the given "pad" value.
    
    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.
    
            - constant: pads with a constant value, this value is specified with fill
    
            - edge: pads with the last value at the edge of the image
    
            - reflect: pads with reflection of image without repeating the last value on the edge
    
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
    
            - symmetric: pads with reflection of image repeating the last value on the edge
    
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.Pad
    * invertransforms.util.invertible.Invertible