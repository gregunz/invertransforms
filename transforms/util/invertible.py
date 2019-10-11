from abc import ABC, abstractmethod


class _Invertible(ABC):
    default_invert_args = dict()

    def invert(self, **kwargs):
        if not self._can_invert:
            raise Exception('Can\'t invert-transformed before the former transformation '
                            '(randomness needs to be defined)')

        # keep all default invert args and override them with given kwargs
        kwargs_tmp = self.default_invert_args.copy()
        kwargs_tmp.update(kwargs)
        kwargs = kwargs_tmp

        return self._invert(**kwargs)

    def set_invert_args(self, **kwargs):
        self.default_invert_args = kwargs

    @abstractmethod
    def _invert(self, **kwargs):
        return NotImplemented

    @abstractmethod
    def _can_invert(self):
        raise NotImplementedError


class Invertible(_Invertible, ABC):
    def _can_invert(self):
        return True


class UndefinedInvertible(_Invertible, ABC):
    # can be implemented if we want a transform to be invertible by setting its random/undefined parameters
    def set(self, *args):
        # Exception instead of NotImplementedError because child class complains
        # about its implementation otherwise (PyCharm)
        raise Exception(f'Not implemented by {self.__class__.__name__}')
