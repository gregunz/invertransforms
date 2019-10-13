import invertransforms as T
import invertransforms.util_functions
from invertransforms.util import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestTransformIf(InvertibleTestCase):

    def test_identity(self):
        tf = T.TransformIf(None, condition=False)
        self.assertEqual(tf(self.n), self.n)
        self.assertIsInstance(tf.inverse(), T.Identity)

    def test_not_a_transform(self):
        tf = T.TransformIf(transform=None, condition=True)
        with self.assertRaises(InvertibleError):
            tf.inverse()

    def test_invert(self):
        tf = T.TransformIf(transform=invertransforms.util_functions.ToTensor(), condition=True)
        self.assertIsInstance(tf.inverse(), invertransforms.util_functions.ToPILImage)

    def test_repr(self):
        tf = invertransforms.util_functions.ToPILImage()
        self.assertEqual(repr(tf), repr(T.TransformIf(tf, True)))
