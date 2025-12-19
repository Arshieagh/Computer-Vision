import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

# CIFAR-100 fine-to-coarse mapping
CIFAR100_FINE_TO_COARSE = [
    4,
    1,
    14,
    8,
    0,
    6,
    7,
    7,
    18,
    3,
    3,
    14,
    9,
    18,
    7,
    11,
    3,
    9,
    7,
    11,
    6,
    11,
    5,
    10,
    7,
    6,
    13,
    15,
    3,
    15,
    0,
    11,
    1,
    10,
    12,
    14,
    16,
    9,
    11,
    5,
    5,
    19,
    8,
    8,
    15,
    13,
    14,
    17,
    18,
    10,
    16,
    4,
    17,
    4,
    2,
    0,
    17,
    4,
    18,
    17,
    10,
    3,
    2,
    12,
    12,
    16,
    12,
    1,
    9,
    19,
    2,
    10,
    0,
    1,
    16,
    12,
    9,
    13,
    15,
    13,
    16,
    19,
    2,
    4,
    6,
    19,
    5,
    5,
    8,
    19,
    18,
    1,
    2,
    15,
    6,
    0,
    17,
    8,
    14,
    13,
]


class CIFAR100WithCoarse(torch.utils.data.Dataset):
    def __init__(self, cifar100_dataset):
        self.dataset = cifar100_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, fine_label = self.dataset[idx]
        coarse_label = CIFAR100_FINE_TO_COARSE[fine_label]
        return image, (fine_label, coarse_label)


# Data loading
def load_data(batch_size=64):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )

    train_dataset = CIFAR100WithCoarse(
        datasets.CIFAR100(
            root="./data",
            train=True,
            transform=train_transform,
            download=True,
        )
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_dataset = CIFAR100WithCoarse(
        datasets.CIFAR100(
            root="./data",
            train=False,
            transform=test_transform,
            download=True,
        )
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_dataloader, test_dataloader
