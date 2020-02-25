import math

import invertransforms as T


def all_crops_resize(size, stride, transform=None, return_unique=False):
    def to(img):
        w, h = img.size
        w = math.floor(w / stride) * stride
        h = math.floor(h / stride) * stride
        img = T.Resize(size=(h, w))(img)
        tfs = []
        unique_crops = []

        for i in range(0, h - size + 1, stride):
            for j in range(0, w - size + 1, stride):
                tf = T.Crop(location=(i, j), size=size)
                if transform is not None:
                    tf = T.Compose([tf, transform])
                tfs += [tf]
                unique_crops += [i % size == 0 and j % size == 0]

        out = [tf(img) for tf in tfs]

        if return_unique:
            return out, unique_crops
        return out

    return to


def all_crops_overlap(size, stride, transform=None, margin=0):
    def to(img):
        w, h = img.size
        tfs = []

        for i in range(0, h - margin, stride):
            for j in range(0, w - margin, stride):
                i = min(i, h - size)
                j = min(j, w - size)
                tf = T.Crop(location=(i, j), size=size)
                if transform is not None:
                    tf = T.Compose([tf, transform])
                tfs += [tf]

        out = [tf(img) for tf in tfs]

        return out

    return to


def list_crops_overlap(h, w, size, stride, margin=0):
    tfs = []
    for i in range(0, h - margin, stride):
        for j in range(0, w - margin, stride):
            i = min(i, h - size)
            j = min(j, w - size)
            tf = T.Crop(location=(i, j), size=size)
            tfs += [tf]
    return tfs
