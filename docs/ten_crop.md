Module invertransforms.ten_crop
===============================

Classes
-------

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