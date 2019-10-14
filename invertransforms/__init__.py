"""
Root module.

This modules contains all the transformations classes.

There are two typical practices to import them into your project:

```python
import invertransforms as T

transform = T.Normalize()
```

```python
from invertransforms import Normalize

transform = Normalize()
```

"""
from torchvision.transforms import functional

from .affine import Affine, RandomAffine, RandomRotation, Rotation
from .color import ColorJitter, Grayscale, RandomGrayscale
from .crop_pad import Crop, RandomCrop, Pad, FiveCrop, TenCrop, CenterCrop
from .perpective import Perspective, RandomPerspective, RandomHorizontalFlip, RandomVerticalFlip
from .resize import Resize, Scale, RandomResizedCrop, RandomSizedCrop
from .sequence import Compose, RandomOrder, RandomChoice, RandomApply
from .tensors import LinearTransformation, Normalize, RandomErasing
from .util_functions import Identity, Lambda, TransformIf, ToPILImage, ToTensor

__all__ = ['Affine', 'CenterCrop', 'ColorJitter', 'Compose', 'Crop', 'FiveCrop', 'Grayscale', 'Identity', 'Lambda',
           'LinearTransformation', 'Normalize', 'Pad', 'Perspective', 'RandomAffine', 'RandomApply', 'RandomChoice',
           'RandomCrop', 'RandomErasing', 'RandomGrayscale', 'RandomHorizontalFlip', 'RandomOrder',
           'RandomPerspective', 'RandomResizedCrop', 'RandomRotation', 'RandomSizedCrop', 'RandomVerticalFlip',
           'Resize', 'Rotation', 'Scale', 'TenCrop', 'ToPILImage', 'ToTensor', 'TransformIf', 'functional']
