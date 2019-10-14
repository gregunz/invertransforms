Module invertransforms.crop_pad
===============================
Crop and Pad module.

This modules contains multiple transformations about creating crops.
Generally, their inverse is/or involves `Pad`, and respectively is `Crop` for `Pad` transformation.

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

`Crop(location, size)`
:   

    ### Ancestors (in MRO)

    * invertransforms.util.invertible.Invertible

`FiveCrop(size)`
:   Crop the given PIL Image into four corners and the central crop
    
    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.
    
    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
    
    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.FiveCrop
    * invertransforms.util.invertible.Invertible

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

`TenCrop(size, vertical_flip=False)`
:   Crop the given PIL Image into four corners and the central crop plus the flipped version of
    these (horizontal flipping is used by default)
    
    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.
    
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        vertical_flip (bool): Use vertical flipping instead of horizontal
    
    Example:
         >>> transform = Compose([
         >>>    TenCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.TenCrop
    * invertransforms.util.invertible.Invertible