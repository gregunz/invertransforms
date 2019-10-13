import invertransforms as T
from invertransforms.util import InvertibleException
from tests.invertible_test_case import InvertibleTestCase


class TestCenterCrop(InvertibleTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tf = T.CenterCrop(self.crop_size)

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleException):
            self.tf.invert()

    def test_inverse(self):
        img = self.tf(self.img_pil)
        self.assertEqual(img.size[::-1], self.crop_size)  # [::-1] because pil order is inversed

        img2 = self.tf.inverse(img)
        self.assertEqual(self.img_pil.size, img2.size)
