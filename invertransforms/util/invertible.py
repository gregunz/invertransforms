from abc import ABC, abstractmethod


class Invertible(ABC):
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

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class InvertibleException(Exception):
    def __init__(self, message):
        super().__init__(message)
