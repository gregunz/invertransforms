import invertransforms as T
from invertransforms.util import InvertibleException
from tests.invertible_test_case import InvertibleTestCase


class TestResize(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleException):
            T.Resize(size=self.crop_size).invert()

    def test_deprecated(self):
        with self.assertWarns(UserWarning):
            T.Scale(size=self.crop_size)
