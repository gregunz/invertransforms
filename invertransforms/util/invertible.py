from abc import ABC, abstractmethod


class _Invertible(ABC):
    __default_invert_args = dict()

    def invert(self, **kwargs):
        if not self.__still_need_forward:
            raise InvertibleException('Can\'t invert-transformed before the former transformation '
                                      '(randomness needs to be defined)')

        # keep all default invert args and override them with given kwargs
        kwargs_tmp = self.__default_invert_args.copy()
        kwargs_tmp.update(kwargs)
        kwargs = kwargs_tmp

        return self._invert(**kwargs)

    @abstractmethod
    def _invert(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _can_invert(self):
        raise NotImplementedError

    def __still_need_forward(self):
        return not self._can_invert()


class Invertible(_Invertible, ABC):
    def _can_invert(self):
        return True


class UndefinedInvertible(_Invertible, ABC):
    # can be implemented if we want a transform to be invertible by setting its random/undefined parameters
    def set(self, *args, **kwargs):
        # Exception instead of NotImplementedError because child class complains
        # about its implementation otherwise (PyCharm)
        raise Exception(f'Not implemented by {self.__class__.__name__}')


class InvertibleException(Exception):
    def __init__(self, message):
        super().__init__(message)
