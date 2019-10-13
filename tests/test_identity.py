import invertransforms as T
from tests.invertible_test_case import InvertibleTestCase


class TestIdentity(InvertibleTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tf = T.Identity()

    def test_call(self):
        self.assertEqual(self.n, self.tf(self.n))

    def test_invert_is_identity(self):
        tf_inv = self.tf.invert()
        self.assertEqual(self.n, tf_inv(self.n))
