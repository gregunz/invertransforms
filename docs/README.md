Module invertransforms
======================
This module exports all the transformation classes.

There are two typical practices to import them into your project:

```python
import invertransforms as T

transform = T.Normalize()
```

```python
from invertransforms import Normalize

transform = Normalize()
```

All transformations have an `inverse` transformation attached to it.

```python
inv_transform = transform.inverse()
img_inv = inv_transform(img)
```

If a transformation is random, it is necessary to apply it once before calling `invert` or `inverse()`.
Otherwise it will raise `InvertibleError`.
On the otherhand, `replay` can be called before, it will simply set the randomness on its first call.

One can create its own invertible transforms either by using the
practical `Lambda` class function or by extending the `Invertible` class available
in the `invertransforms.lib` module.

For convenience, you can also import the following torchvision useful functions from this library:
```
# from torchvision.transforms import functional as F
# becomes:

from invertransforms import functional as F
```
Such that more functions could be added in the future.

Sub-modules
-----------
* invertransforms.affine
* invertransforms.color
* invertransforms.crop_pad
* invertransforms.lib
* invertransforms.perpective
* invertransforms.resize
* invertransforms.sequence
* invertransforms.tensors
* invertransforms.util_functions

Classes
-------

`Affine(matrix)`
:   Apply affine transformation on the image.
    
    Args:
        matrix (list of int): transformation matrix (from destination image to source)
         because we want to interpolate the (discrete) destination pixel from the local
         area around the (floating) source pixel.

    ### Ancestors (in MRO)

    * invertransforms.lib.Invertible

`CenterCrop(size)`
:   Crops the given PIL Image at the center.
    
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.CenterCrop
    * invertransforms.lib.Invertible

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

`Compose(transforms)`
:   Composes several transforms together.
    
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.Compose
    * invertransforms.lib.Invertible

`Crop(location, size)`
:   Crop an image.
    Args:
        location (int, tuple): upper left coordinates of the crop area (top, left)
        size (int, tuple): size of the crop (height, width)

    ### Ancestors (in MRO)

    * invertransforms.lib.Invertible

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

`HorizontalFlip(*args, **kwargs)`
:   Flip the image horizontally.

    ### Ancestors (in MRO)

    * invertransforms.lib.Invertible

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

`LinearTransformation(transformation_matrix, mean_vector)`
:   Transform a tensor image with a square transformation matrix and a mean_vector computed
    offline.
    Given transformation_matrix and mean_vector, will flatten the torch.*Tensor and
    subtract mean_vector from it which is then followed by computing the dot
    product with the transformation matrix and then reshaping the tensor to its
    original shape.
    
    Applications:
        whitening transformation: Suppose X is a column vector zero-centered data.
        Then compute the data covariance matrix [D x D] with torch.mm(X.t(), X),
        perform SVD on this matrix and pass it as transformation_matrix.
    
    Args:
        transformation_matrix (Tensor): tensor [D x D], D = C x H x W
        mean_vector (Tensor): tensor [D], D = C x H x W

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.LinearTransformation
    * invertransforms.lib.Invertible

`Normalize(mean, std, inplace=False)`
:   Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.Normalize
    * invertransforms.lib.Invertible

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
    * invertransforms.lib.Invertible

`Perspective(startpoints, endpoints, interpolation=3)`
:   

    ### Ancestors (in MRO)

    * invertransforms.lib.Invertible

`RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)`
:   Random affine transformation of the image keeping center invariant
    
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)
    
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomAffine
    * invertransforms.lib.Invertible

`RandomApply(transforms, p=0.5)`
:   Apply randomly a list of transformations with a given probability
    
    Args:
        transforms: one or multiple of transformations
        p (float): probability

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomApply
    * torchvision.transforms.transforms.RandomTransforms
    * invertransforms.lib.Invertible

`RandomChoice(transforms)`
:   Apply single transformation randomly picked from a list

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomChoice
    * torchvision.transforms.transforms.RandomTransforms
    * invertransforms.lib.Invertible

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
    * invertransforms.lib.Invertible

    ### Methods

    `get_params(self, img, output_size)`
    :   Get parameters for ``crop`` for a random crop.
        
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.

`RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)`
:   Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.
    
    Returns:
        Erased Image.
    # Examples:
        >>> transform = transforms.Compose([
        >>> transforms.RandomHorizontalFlip(),
        >>> transforms.ToTensor(),
        >>> transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> transforms.RandomErasing(),
        >>> ])

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomErasing
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

`RandomHorizontalFlip(p=0.5)`
:   Horizontally flip the given PIL Image randomly with a given probability.
    
    Args:
        p (float): probability of the image being flipped. Default value is 0.5

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomHorizontalFlip
    * invertransforms.lib.Invertible

`RandomOrder(transforms)`
:   Apply a list of transformations in a random order

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomOrder
    * torchvision.transforms.transforms.RandomTransforms
    * invertransforms.lib.Invertible

`RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)`
:   Performs Perspective transformation of the given PIL Image randomly with a given probability.
    
    Args:
        interpolation : Default- Image.BICUBIC
    
        p (float): probability of the image being perspectively transformed. Default value is 0.5
    
        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomPerspective
    * invertransforms.lib.Invertible

`RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)`
:   Crop the given PIL Image to random size and aspect ratio.
    
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomResizedCrop
    * invertransforms.lib.Invertible

    ### Descendants

    * invertransforms.resize.RandomSizedCrop

`RandomRotation(degrees, resample=False, expand=False, center=None)`
:   Rotate the image by angle.
    
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomRotation
    * invertransforms.lib.Invertible

`RandomSizedCrop(*args, **kwargs)`
:   Note: This transform is deprecated in favor of RandomResizedCrop.

    ### Ancestors (in MRO)

    * invertransforms.resize.RandomResizedCrop
    * torchvision.transforms.transforms.RandomResizedCrop
    * invertransforms.lib.Invertible

`RandomVerticalFlip(p=0.5)`
:   Vertically flip the given PIL Image randomly with a given probability.
    
    Args:
        p (float): probability of the image being flipped. Default value is 0.5

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomVerticalFlip
    * invertransforms.lib.Invertible

`Resize(size, interpolation=2)`
:   Resize the input PIL Image to the given size.
    
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.Resize
    * invertransforms.lib.Invertible

    ### Descendants

    * invertransforms.resize.Scale

`Rotation(angle, resample=False, expand=False, center=None)`
:   Rotate the image given an angle (in degrees).
    
    Args:
        angle (float or int): degrees of the angle
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.

    ### Ancestors (in MRO)

    * invertransforms.lib.Invertible

`Scale(*args, **kwargs)`
:   Note: This transform is deprecated in favor of Resize.

    ### Ancestors (in MRO)

    * invertransforms.resize.Resize
    * torchvision.transforms.transforms.Resize
    * invertransforms.lib.Invertible

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

`VerticalFlip(*args, **kwargs)`
:   Flip the image vertically.

    ### Ancestors (in MRO)

    * invertransforms.lib.Invertible