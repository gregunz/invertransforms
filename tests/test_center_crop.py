import invertransforms.crop_pad
from invertransforms.util import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestCenterCrop(InvertibleTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tf = invertransforms.crop_pad.CenterCrop(self.crop_size)

    def test_invert_before_apply(self):
        with self.assertRaises(InvertibleError):
            self.tf.inverse()

    def test_inverse(self):
        img = self.tf(self.img_pil)
        self.assertEqual(img.size[::-1], self.crop_size)  # [::-1] because pil order is inversed

        img2 = self.tf.invert(img)
        self.assertEqual(self.img_pil.size, img2.size)
