import invertransforms as T
from invertransforms.lib import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestRandomFlip(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleError):
            T.RandomVerticalFlip().inverse()

        with self.assertRaises(InvertibleError):
            T.RandomHorizontalFlip().inverse()
