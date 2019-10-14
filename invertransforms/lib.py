"""
This module contains the basic building blocks of this library.
It contains the abstract class all transformations should extend
and utility functions.

"""
import random
from abc import abstractmethod


class Invertible:
    _tracked_inverses = dict()

    @abstractmethod
    def __call__(self, img):
        """
        Apply the transformation

        Args:
            img (PIL Image, torch.Tensor, Any): input image

        Returns (PIL Image, torch.Tensor, Any): transformed input

        """
        raise NotImplementedError

    @abstractmethod
    def inverse(self) -> 'Invertible':
        """
        Abstract method to return the inverse of the transformation

        Returns (Invertible): tf

        """
        raise NotImplementedError

    def track(self, img, index=None):
        """
        Apply the transformation and track all inverses.

        Args:
            img (PIL Image, torch.Tensor, Any): input image.
            index (optional, int or Any): index associated with the tracked inverse transform;
             increasing int when not defined

        Returns: image
        """
        if index is None:
            index = len(self._tracked_inverses)
        img = self.__call__(img)
        self._tracked_inverses[index] = self.inverse()
        return img

    def get_inverse(self, index) -> 'Invertible':
        """
        Get the inverse of a tracked transformation given its index.

        Args:
            index (int or Any): index associated with the tracked inverse transform

        Returns:
            inverse transformation
        """
        return self._tracked_inverses[index]

    def __getitem__(self, index):
        return self.get_inverse(index)

    def invert(self, img):
        """
        Apply the inverse of this transformation.

        Args:
            img (PIL Image, torch.Tensor, Any): input image

        Returns: image

        """
        return self.inverse()(img)

    def replay(self, img):
        """
        Replay a transformation (with random like previous runs).
        If it is called before any calls to `__call__`, it will simply calls `__call__`

        Note: Any call to `__call__` will change the randomness again.

        Args:
            img (PIL Image, torch.Tensor, Any): input image

        Returns: image

        """
        try:
            # hack: because inverse fixes the randomness,
            #       we can replay for free with double inverse
            return self.inverse().inverse()(img)
        except InvertibleError:
            return self.__call__(img)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    # not very useful except for refactoring
    # def _can_invert(self):
    #    return True


class InvertibleError(Exception):
    """
    Error raised when transformation cannot be inverted.
    """

    def __init__(self, message):
        super().__init__(message)


def flip_coin(p):
    """
    Return true with probability p

    Args:
        p: float, probability to return True

    Returns: bool

    """

    assert 0 <= p <= 1, 'A probability should be between 0 and 1'
    return random.random() < p
