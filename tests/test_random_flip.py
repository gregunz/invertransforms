import invertransforms as T
from invertransforms.util import InvertibleException
from tests.invertible_test_case import InvertibleTestCase


class TestRandomFlip(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleException):
            T.RandomVerticalFlip().invert()

        with self.assertRaises(InvertibleException):
            T.RandomHorizontalFlip().invert()
