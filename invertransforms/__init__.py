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

# noinspection PyUnresolvedReferences
from torchvision.transforms import functional

from invertransforms.color_transforms import Grayscale, RandomGrayscale
from invertransforms.crop_pad import FiveCrop, TenCrop, CenterCrop
from invertransforms.list_transforms import Compose
from invertransforms.random_erasing import RandomErasing
from invertransforms.tensor_transforms import Normalize
from invertransforms.util_functions import ToPILImage, ToTensor
from .affine import Affine, RandomAffine
from .color_transforms import ColorJitter
from .crop_pad import Crop, RandomCrop, Pad
from .flip import RandomHorizontalFlip, RandomVerticalFlip
from .list_transforms import RandomOrder, RandomChoice, RandomApply
from .perpective import Perspective, RandomPerspective
from .random_resized_crop import RandomResizedCrop, RandomSizedCrop
from .resize import Resize, Scale
from .rotation import RandomRotation, Rotation
from .tensor_transforms import LinearTransformation
from .util_functions import Identity, Lambda, TransformIf

__all__ = [
    'functional', 'Affine', 'RandomAffine', 'ColorJitter',
    'Crop', 'RandomCrop', 'Identity', 'Lambda', 'LinearTransformation',
    'Pad', 'Perspective', 'RandomPerspective', 'RandomHorizontalFlip', 'RandomVerticalFlip',
    'RandomResizedCrop',
    'RandomChoice', 'RandomApply', 'Resize', 'Scale', 'RandomRotation',
    'Rotation', 'TransformIf',
    'RandomSizedCrop', 'RandomOrder'
]
