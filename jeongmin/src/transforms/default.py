from .base import BaseTransforms

import torchvision.transforms as T


class DefaultTransforms(BaseTransforms):
    def __init__(self) -> None:
        super().__init__()
        # CIFAR-10 statistics
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2470, 0.2435, 0.2616]

    def train_transform(self):
        return T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.RandAugment(num_ops=2, magnitude=9),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

    def val_transform(self):
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

    def test_transform(self):
        return self.val_transform()
