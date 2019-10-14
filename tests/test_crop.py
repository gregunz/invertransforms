import invertransforms as T
from invertransforms.lib import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestCrop(InvertibleTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.location = self.n
        self.tf = T.Crop(location=self.location, size=self.crop_size)
        self.random_tf = T.RandomCrop(self.crop_size)

    def test_constructor(self):
        tf2 = T.Crop(location=(self.location, self.location), size=self.crop_size[0])

        self.assertEqual(self.tf.tl_i, tf2.tl_i)
        self.assertEqual(self.tf.tl_j, tf2.tl_j)
        self.assertEqual(self.tf.crop_h, tf2.crop_h)
        self.assertEqual(self.tf.crop_w, self.crop_size[1])

        with self.assertRaises(Exception):
            T.Crop(location=(0, 0, 0), size=self.crop_size)

        with self.assertRaises(Exception):
            T.Crop(location=self.location, size=None)

    def test_rep(self):
        self.assertTrue('Crop' in repr(self.tf))
        self.assertTrue(str(self.location) in repr(self.tf))
        self.assertTrue(str(self.crop_size) in repr(self.tf))

    def test_call(self):
        self.assertEqual(self.tf(self.img_pil).size[::-1], self.crop_size)  # [::-1] because pil order is inversed
        self.assertEqual(self.random_tf(self.img_pil).size[::-1], self.crop_size)

    def test_invert_before_applied(self):
        with self.assertRaises(InvertibleError):
            self.tf.inverse()
        with self.assertRaises(InvertibleError):
            self.random_tf.inverse()

    def test_invert(self):
        img_tf = self.tf(self.img_pil)
        # one could check in a better way whether inverse is correct
        self.assertEqual(self.tf.invert(img_tf).size, self.img_pil.size)

        img_random_tf = self.random_tf(self.img_pil)
        self.assertEqual(self.random_tf.invert(img_random_tf).size, self.img_pil.size)
