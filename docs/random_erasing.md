Module invertransforms.random_erasing
=====================================

Classes
-------

`RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)`
:   Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.
    
    Returns:
        Erased Image.
    # Examples:
        >>> transform = transforms.Compose([
        >>> transforms.RandomHorizontalFlip(),
        >>> transforms.ToTensor(),
        >>> transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> transforms.RandomErasing(),
        >>> ])

    ### Ancestors (in MRO)

    * torchvision.transforms.transforms.RandomErasing
    * invertransforms.util.invertible.Invertible