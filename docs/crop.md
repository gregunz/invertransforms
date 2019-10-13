Module invertransforms.crop
===========================

Classes
-------

`Crop(location, size)`
:   

    ### Ancestors (in MRO)

    * invertransforms.util.invertible.Invertible

`RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')`
:   Crop the given PIL Image at a random location.
    
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
    
             - constant: pads with a constant value, this value is specified with fill
    
             - edge: pads with the last value on the edge of the image
    
             - reflect: pads with reflection of image (without repeating the last value on the edge)
    
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
    
             - symmetric: pads with reflection of image (repeating the last value on the edge)
    
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomCrop
    * invertransforms.util.invertible.Invertible

    ### Methods

    `get_params(self, img, output_size)`
    :   Get parameters for ``crop`` for a random crop.
        
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.