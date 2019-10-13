import invertransforms as T
from invertransforms.util import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestRandomFlip(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleError):
            T.RandomVerticalFlip().invert()

        with self.assertRaises(InvertibleError):
            T.RandomHorizontalFlip().invert()
