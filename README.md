[Documentation]: https://github.com/gregunz/invertransforms

[![](https://i.imgur.com/dFDH5Ro.jpg)](https://github.com/gregunz/invertransforms)

invertransforms
====

[![Build Status](https://img.shields.io/travis/gregunz/invertransforms.svg?style=for-the-badge)](https://travis-ci.org/gregunz/invertransforms)
[![Code Coverage](https://img.shields.io/codecov/c/gh/gregunz/invertransforms.svg?style=for-the-badge)](https://codecov.io/gh/gregunz/invertransforms)
[![pdoc3 on PyPI](https://img.shields.io/pypi/v/invertransforms.svg?color=blue&style=for-the-badge)](https://pypi.org/project/invertransforms)

A library which turns torchvision transformations __invertible__ and __replayable__.


Installation
------------
```bash
pip install invertransforms
```

Usage
-----
Simply replace previous torchvision import statements and enjoy the addons.

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
```
All transformations have an `inverse` transformation attached to it.


```python
inv_transform = transform.inverse()
img_inv = inv_transform(img)
```
__Notes:__

If a transformation is random, it is necessary to apply it once before calling `invert` or `inverse()`. Otherwise it will raise `InvertibleError`. 
On the otherhand, `replay` can be called before, it will simply set the randomness on its first call.


Features
--------
* Invert any transformations even random ones
* Replay any transformations even random ones
* Extending torchvision tranformations
* blah-blah


[Documentation]
-------------

The above features are explained in more detail in invertransforms' [documentation].
