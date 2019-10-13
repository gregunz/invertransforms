import invertransforms.crop_pad
from invertransforms.util import InvertibleError
from tests.invertible_test_case import InvertibleTestCase


class TestTenCrop(InvertibleTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tf = invertransforms.crop_pad.TenCrop(size=self.crop_size)

    def test_call(self):
        outputs = self.tf(self.img_pil)
        self.assertEqual(len(outputs), 10)

        for img in outputs:
            self.assertEqual(img.size[::-1], self.crop_size)  # [::-1] because pil order is inversed

    def test_call_vflip(self):
        tf = invertransforms.crop_pad.TenCrop(self.crop_size, vertical_flip=True)
        outputs = tf(self.img_pil)
        self.assertEqual(len(outputs), 10)

        for img in outputs:
            self.assertEqual(img.size[::-1], self.crop_size)  # [::-1] because pil order is inversed

    def test_invert_before_call(self):
        with self.assertRaises(InvertibleError):
            self.tf.inverse()

    def test_invert(self):
        outputs = self.tf(self.img_pil)
        outputs_inv = self.tf.invert(outputs)

        for i, img in enumerate(outputs_inv):
            self.assertEqual(img.size, self.img_pil.size, msg=f'output {i}')

    def test_invert_invert(self):
        outputs = self.tf(self.img_pil)
        tf_inv = self.tf.inverse()
        tf_inv_inv = self.tf.inverse().inverse()

        outputs_inv = tf_inv(outputs)
        outputs_bis = tf_inv_inv(outputs_inv)

        for img in outputs_bis:
            self.assertEqual(img.size[::-1], self.crop_size)  # [::-1] because pil order is inversed
