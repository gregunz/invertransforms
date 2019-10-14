"""
Root module.

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

You can create your own invertible transformation class either by using the
practical `Lambda` class function or by extending the `Invertible` class available
in the `invertransforms.lib` module.


For conveniences, you can also import the following torchvision useful functions from this library:
```
# from torchvision.transforms import functional as F
# becomes:

from invertransforms import functional as F
```
"""
from torchvision.transforms import functional

from .affine import Affine, RandomAffine, RandomRotation, Rotation
from .color import ColorJitter, Grayscale, RandomGrayscale
from .crop_pad import Crop, RandomCrop, Pad, FiveCrop, TenCrop, CenterCrop
from .perpective import Perspective, VerticalFlip, HorizontalFlip, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomPerspective
from .resize import Resize, Scale, RandomResizedCrop, RandomSizedCrop
from .sequence import Compose, RandomOrder, RandomChoice, RandomApply
from .tensors import LinearTransformation, Normalize, RandomErasing
from .util_functions import Identity, Lambda, TransformIf, ToPILImage, ToTensor

__all__ = ['Affine', 'CenterCrop', 'ColorJitter', 'Compose', 'Crop', 'FiveCrop', 'Grayscale', 'Identity', 'Lambda',
           'LinearTransformation', 'Normalize', 'Pad', 'Perspective', 'RandomAffine', 'HorizontalFlip', 'VerticalFlip',
           'RandomApply', 'RandomChoice', 'RandomCrop', 'RandomErasing', 'RandomGrayscale', 'RandomHorizontalFlip',
           'RandomOrder', 'RandomPerspective', 'RandomResizedCrop', 'RandomRotation', 'RandomSizedCrop',
           'RandomVerticalFlip', 'Resize', 'Rotation', 'Scale', 'TenCrop', 'ToPILImage', 'ToTensor', 'TransformIf',
           'functional']
