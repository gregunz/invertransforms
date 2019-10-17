from typing import List

from torchvision.transforms import transforms

import invertransforms as T
from invertransforms.lib import Invertible, InvertibleError


def extract_transforms(transform: Invertible, filter_random=True, filter_identity=True) -> List[Invertible]:
    if filter_random and not isinstance(transform, Invertible):
        raise ValueError('This function is only defined over Invertible transforms when filter_random is True')

    if isinstance(transform, T.TransformIf):
        transform = transform.transform

    # transforms that might not be applied when called:
    # rdm_tf = (T.RandomChoice, T.RandomApply, T.RandomOrder, T.RandomHorizontalFlip, T.RandomVerticalFlip,
    #          T.RandomGrayscale, T.RandomErasing, T.Random)
    if filter_random:  # and isinstance(transform, rdm_tf):
        try:
            transform = transform.inverse().inverse()
        except InvertibleError:
            raise InvertibleError(f'Cannot extract {transform.__class__.__name__} before it has been applied.')

    # only RandomTransforms and Compose contain multiple transforms
    if isinstance(transform, transforms.RandomTransforms) or isinstance(transform, transforms.Compose):
        list_transforms = []
        for tf in transform.transforms:
            list_transforms += extract_transforms(tf, filter_random=filter_random, filter_identity=filter_identity)
        return list_transforms

    if filter_identity and isinstance(transform, T.Identity):
        return []

    return [transform]
