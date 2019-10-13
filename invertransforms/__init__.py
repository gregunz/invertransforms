"""
Root module.

This modules contains all the transformations classes.

There are two typical practices to import them into your project:

```python
import invertransform as T

transform = T.Normalize()
```

```python
from invertransform import Normalize

transform = Normalize()
```

"""

# noinspection PyUnresolvedReferences
from torchvision.transforms import functional

from .affine import Affine, RandomAffine
from .center_crop import CenterCrop
from .color_jitter import ColorJitter
from .compose import Compose
from .crop import Crop, RandomCrop
from .five_crop import FiveCrop
from .functions import Identity, Lambda, TransformIf
from .grayscale import Grayscale, RandomGrayscale
from .linear_transformation import LinearTransformation
from .normalize import Normalize
from .pad import Pad
from .perpective import Perspective, RandomPerspective
from .random_erasing import RandomErasing
from .random_flip import RandomHorizontalFlip, RandomVerticalFlip
from .random_resized_crop import RandomResizedCrop, RandomSizedCrop
from .random_transforms import RandomOrder, RandomChoice, RandomApply
from .resize import Resize, Scale
from .rotation import RandomRotation, Rotation
from .ten_crop import TenCrop
from .to_tensor_pil_image import ToPILImage, ToTensor

__all__ = [
    'functional', 'Affine', 'RandomAffine', 'CenterCrop', 'ColorJitter',
    'Compose', 'Crop', 'RandomCrop', 'FiveCrop', 'Grayscale',
    'RandomGrayscale', 'Identity', 'Lambda', 'LinearTransformation',
    'Normalize', 'Pad', 'Perspective', 'RandomPerspective', 'RandomErasing',
    'RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomResizedCrop',
    'RandomChoice', 'RandomApply', 'Resize', 'Scale', 'RandomRotation',
    'Rotation', 'TenCrop', 'ToPILImage', 'ToTensor', 'TransformIf',
    'RandomSizedCrop', 'RandomOrder'
]
