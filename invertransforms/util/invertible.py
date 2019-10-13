from abc import ABC, abstractmethod


class Invertible(ABC):

    @abstractmethod
    def __call__(self, img):
        raise NotImplementedError

    @abstractmethod
    def invert(self):
        raise NotImplementedError

    def inverse(self, img):
        """
        Apply inverse transformation

        :param img:
        :return:
        """
        return self.invert()(img)

    def replay(self, img):
        """
        Replay a transformation (with random like previous runs).
        If it is called before any calls to __call__, it will simply calls __call__

        Note: Any call to __call__ will change the randomness again.

        :param img:
        :return:
        """
        try:
            # hack: because inverse fixes the randomness,
            #       we can replay for free with double inverse
            return self.invert().invert().__call__(img)
        except InvertibleException:
            return self.__call__(img)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    # not very useful except for refactoring
    def _can_invert(self):
        return True


class InvertibleException(Exception):
    def __init__(self, message):
        super().__init__(message)
