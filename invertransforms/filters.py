from torchvision.transforms import transforms

import invertransforms as T
from invertransforms.lib import Invertible, InvertibleError


def extract_transforms(transform: Invertible, only_if_applied=True, filter_passive=True):
    if not isinstance(transform, Invertible):
        raise ValueError('this function is defined over Invertible transforms')

    # transforms that might not be applied when called:
    # rdm_tf = (T.RandomChoice, T.RandomApply, T.RandomOrder, T.RandomHorizontalFlip, T.RandomVerticalFlip,
    #          T.RandomGrayscale, T.RandomErasing, T.Random)
    if only_if_applied:  # and isinstance(transform, rdm_tf):
        try:
            transform = transform.inverse().inverse()
        except InvertibleError:
            raise InvertibleError(f'Cannot extract {transform.__class__.__name__} before it has been applied.')
    elif filter_passive and isinstance(transform, T.TransformIf):
        transform = transform.transform

    # only torchvision RandomTransforms and Compose contain multiple transforms
    if isinstance(transform, transforms.RandomTransforms) or isinstance(transform, T.Compose):
        list_transforms = []
        for tf in transform.transforms:
            list_transforms += extract_transforms(tf, only_if_applied=only_if_applied, filter_passive=filter_passive)
        return list_transforms

    if filter_passive and isinstance(transform, T.Identity):
        return []

    return [transform]
