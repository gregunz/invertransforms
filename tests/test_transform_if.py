import invertransforms as T
from invertransforms.util import InvertibleException
from tests.invertible_test_case import InvertibleTestCase


class TestTransformIf(InvertibleTestCase):

    def test_identity(self):
        tf = T.TransformIf(None, condition=False)
        self.assertEqual(tf(self.n), self.n)
        self.assertIsInstance(tf.invert(), T.Identity)

    def test_not_a_transform(self):
        tf = T.TransformIf(transform=None, condition=True)
        with self.assertRaises(InvertibleException):
            tf.invert()

    def test_invert(self):
        tf = T.TransformIf(transform=T.ToTensor(), condition=True)
        self.assertIsInstance(tf.invert(), T.ToPILImage)

    def test_repr(self):
        tf = T.ToPILImage()
        self.assertEqual(repr(tf), repr(T.TransformIf(tf, True)))
