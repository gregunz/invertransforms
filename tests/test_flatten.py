import invertransforms as T
from invertransforms.extract import extract_transforms
from tests.invertible_test_case import InvertibleTestCase


class TestCrop(InvertibleTestCase):
    def setUp(self) -> None:
        super().setUp()

        identity_block = T.Compose([
            T.ToTensor(),
            T.Lambda(lambd=lambda x: 2 * x, tf_inv=lambda x: x / 2, repr_str='TimesTwo()'),
            T.RandomErasing(),
            T.Lambda(lambd=lambda x: 2 * x, tf_inv=lambda x: x / 2, repr_str='TimesTwo()').inverse(),
            T.ToPILImage(),
        ])

        self.crazy_tf = T.Compose([
            T.RandomOrder([
                T.TransformIf(T.RandomHorizontalFlip(), True),
                T.Compose([
                    T.RandomChoice([
                        T.RandomApply([
                            T.RandomCrop(self.crop_size)
                        ])
                    ])
                ]),
                T.Identity(log_fn=lambda img: print(img.size)),
                identity_block,
            ]),
            T.ToTensor(),
            T.RandomChoice([
                T.Normalize(mean=[0.5], std=[0.5]),
                T.Normalize(mean=[0.3], std=[0.3]),
            ]),
        ])

    def test_flatten_nested(self):
        print(self.crazy_tf)
        all_tf = extract_transforms(self.crazy_tf, filter_random=False)
        print(all_tf)

        self.crazy_tf(self.img_pil)
        only_applied_tf = extract_transforms(self.crazy_tf, filter_random=True)
        print(only_applied_tf)
