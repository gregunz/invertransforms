Module invertransforms.normalize
================================

Classes
-------

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
    * invertransforms.util.invertible.Invertible