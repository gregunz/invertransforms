import invertransforms as T
from invertransforms.util import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestPerspective(InvertibleTestCase):

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleError):
            T.RandomPerspective().inverse()

    def test_invert(self):
        tf_random = T.RandomPerspective(p=1)
        img_inv = tf_random(self.img_pil)
        tf_inv = tf_random.inverse()
        # inversion is not pixel perfect so we are only comparing size for now
        self.assertEqual(tf_inv(img_inv).size, self.img_pil.size)

        self.assertIn('Perspective', repr(tf_inv))
        self.assertIn('startpoints=', repr(tf_inv))
        self.assertIn('endpoints=', repr(tf_inv))
        self.assertIsInstance(tf_inv, T.Perspective)
        self.assertIsInstance(tf_inv.inverse(), T.Perspective)

    def test_identity(self):
        tf_id = T.RandomPerspective(p=0)
        self.assertEqual(self.n, tf_id(self.n))
        self.assertEqual(self.n, tf_id.invert(self.n))
