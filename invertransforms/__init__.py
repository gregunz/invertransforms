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

from .affine import Affine, RandomAffine
from .color_transforms import ColorJitter
from .color_transforms import Grayscale, RandomGrayscale
from .crop_pad import Crop, RandomCrop, Pad
from .crop_pad import FiveCrop, TenCrop, CenterCrop
from .flip import RandomHorizontalFlip, RandomVerticalFlip
from .list_transforms import Compose
from .list_transforms import RandomOrder, RandomChoice, RandomApply
from .perpective import Perspective, RandomPerspective
from .random_erasing import RandomErasing
from .random_resized_crop import RandomResizedCrop, RandomSizedCrop
from .resize import Resize, Scale
from .rotation import RandomRotation, Rotation
from .tensor_transforms import LinearTransformation
from .tensor_transforms import Normalize
from .util_functions import Identity, Lambda, TransformIf
from .util_functions import ToPILImage, ToTensor

__all__ = ['Affine', 'CenterCrop', 'ColorJitter', 'Compose', 'Crop', 'FiveCrop', 'Grayscale', 'Identity', 'Lambda',
           'LinearTransformation', 'Normalize', 'Pad', 'Perspective', 'RandomAffine', 'RandomApply', 'RandomChoice',
           'RandomCrop', 'RandomErasing', 'RandomGrayscale', 'RandomHorizontalFlip', 'RandomOrder',
           'RandomPerspective', 'RandomResizedCrop', 'RandomRotation', 'RandomSizedCrop', 'RandomVerticalFlip',
           'Resize', 'Rotation', 'Scale', 'TenCrop', 'ToPILImage', 'ToTensor', 'TransformIf', 'functional']
