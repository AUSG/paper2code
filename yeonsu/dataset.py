import torch
import torchvision
import torchvision.transforms as transforms
from augmentation import get_advanced_augmentation


def get_cifar10_dataloaders(data_dir='./data', batch_size=128, num_workers=4, 
                            use_advanced_aug=False, use_cutout=True, cutout_length=16):
    if use_advanced_aug:
        train_transform, test_transform = get_advanced_augmentation(
            use_cutout=use_cutout, 
            cutout_length=cutout_length
        )
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes
