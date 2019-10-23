[Documentation]: https:///gregunz.github.io/invertransforms/
[mail@gregunz.io]: mailto:mail@gregunz.io

[![](https://i.imgur.com/dFDH5Ro.jpg)](https://github.com/gregunz/invertransforms)

invertransforms
====

[![Build Status](https://img.shields.io/travis/com/gregunz/invertransforms.svg?style=for-the-badge)](https://travis-ci.com/gregunz/invertransforms)
[![Code Coverage](https://img.shields.io/codecov/c/gh/gregunz/invertransforms?style=for-the-badge)](https://codecov.io/gh/gregunz/invertransforms)
[![PyPI](https://img.shields.io/pypi/v/invertransforms.svg?color=blue&style=for-the-badge)](https://pypi.org/project/invertransforms)

A library which turns torchvision transformations __invertible__ and __replayable__.


Installation
------------
```bash
pip install invertransforms
```

Usage
-----
Simply replace previous torchvision import statements and enjoy the new features.

```python
# from torchvision import transforms as T
import invertransforms as T

transform = T.Compose([
  T.RandomCrop(size=256),
  T.ToTensor(),
])

img_tensor = transform(img)

# invert
img_again = transform.invert(img_tensor)

# replay
img_tensor2 = transform.replay(img2)

# track
for i in range(n):
    img_tensor_i = transform.track(img_i)
    # ...
inverse_tf = transform.get_inverse(j)  # or transform[j]
img_j = inverse_tf(img_tensor_j)
```

All transformations have an `inverse` transformation attached to it.

```python
inv_transform = transform.inverse()
img_inv = inv_transform(img)
```
__Notes:__

If a transformation is random, it is necessary to apply it once before calling `invert` or `inverse()`. Otherwise it will raise `InvertibleError`. 
On the otherhand, `replay` can be called before, it will simply set the randomness on its first call.


One can create its own invertible transforms either by using the
practical `Lambda` class function or by extending the `Invertible` class available
in the `invertransforms.lib` module.


[Documentation]
---------------

The library's [documentation] contains the full list of [transformations](https://gregunz.github.io/invertransforms/#header-classes)
 which includes all the ones from torchvision and more.

Use Cases
---------
This library can be particularly useful in following situations:

- Reverting a NN-model output in order to stack predictions

- Applying the same (random) transformations the same way on different inputs

Features
--------
- Invert any transformations, even random ones

- Replay any transformations, even random ones

- Track all transformations to invert them long after

- All classes extend its torchvision transformation equivalent class.
 This means, you can just replace your previous torchvision import statements and it will not break your code.
 
- Extensive unit testing (100% coverage, be safe, hopefully)

Requirements
------------
```
python>=3.6

torch>=1.2.0
torchvision>=0.4.0
```


Future Improvements
-------------------
- [WIP] Extend the number of tranformations (e.g. random rotation and cropping (within the rotated area))

- Make the transformations on tensors directly (data augmentation/transformation on GPU)


Contribute
----------
You found a bug, think a feature is missing or just want to help ?

Please feel free to open an issue, pull request or contact me [mail@gregunz.io]

