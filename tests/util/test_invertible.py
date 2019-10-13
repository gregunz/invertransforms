import unittest

from invertransforms.util import Invertible


class TestInvertible(unittest.TestCase):

    def test_class(self):
        class NewInvertible(Invertible):
            pass

        with self.assertRaises(TypeError):
            NewInvertible()

        # This test is used for accessing abstract method
        class NewInvertibleBis(Invertible):
            def invert(self):
                super().invert()

        tf = NewInvertibleBis()
        with self.assertRaises(NotImplementedError):
            tf.invert()

        self.assertIn(tf.__class__.__name__, repr(tf))
