import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.Hyperparameters import Hyperparameters as hp
import numpy as np

class DataLoaderWrapper:
    def __init__(self):
        if hp.dataset == 'CIFAR10':
            self.train_loader, self.val_loader, self.test_loader = self.get_cifar10_loaders()
        elif hp.dataset == 'ImageNet':
            self.train_loader, self.val_loader = self.get_imagenet_loaders()
        else:
            raise ValueError("Invalid dataset. Choose 'CIFAR10' or 'ImageNet'.")

    def get_cifar10_loaders(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        # Print the number of samples in the dataset
        print('Number of training samples:', len(train_set))
        test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        # Print the number of samples in the dataset
        print('Number of test samples:', len(test_set))
        
        #print the data shape
        print('training Data shape:', train_set[0][0].shape)
        #print the data shape
        print('test Data shape:', test_set[0][0].shape)

        # Split train_set into training and validation sets
        train_size = int(0.8 * len(train_set))
        val_size = len(train_set) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_set, [train_size, val_size], generator=torch.Generator().manual_seed(hp.seed))

        if hp.use_noise or hp.use_occlusion:
            train_dataset = self.apply_augmentations(train_dataset)
            val_dataset = self.apply_augmentations(val_dataset)
            test_set = self.apply_augmentations(test_set)

        train_loader = DataLoader(
            train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(
            val_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(
            test_set, batch_size=hp.batch_size, shuffle=False, num_workers=2)

        return train_loader, val_loader, test_loader

    def get_imagenet_loaders(self):
        # You need to specify the path to your ImageNet dataset
        data_dir = '/path/to/imagenet'
        train_dir = f'{data_dir}/train'
        val_dir = f'{data_dir}/val'

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            ),
        ])

        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            ),
        ])

        train_dataset = torchvision.datasets.ImageFolder(
            root=train_dir, transform=transform_train)
        val_dataset = torchvision.datasets.ImageFolder(
            root=val_dir, transform=transform_val)

        if hp.use_noise or hp.use_occlusion:
            train_dataset = self.apply_augmentations(train_dataset)
            val_dataset = self.apply_augmentations(val_dataset)

        train_loader = DataLoader(
            train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(
            val_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, val_loader

    def apply_augmentations(self, dataset):
        # Apply noise or occlusion to the dataset
        augmented_data = []
        for img, label in dataset:
            if hp.use_noise:
                img = self.add_noise(img)
            if hp.use_occlusion:
                img = self.add_occlusion(img)
            augmented_data.append((img, label))
        return augmented_data

    def add_noise(self, img):
        noise = torch.randn_like(img) * 0.1
        img_noisy = img + noise
        img_noisy = torch.clamp(img_noisy, 0., 1.)
        return img_noisy

    def add_occlusion(self, img):
        c, h, w = img.size()
        occlusion_size = int(h * 0.2)
        x = np.random.randint(0, h - occlusion_size)
        y = np.random.randint(0, w - occlusion_size)
        img[:, x:x+occlusion_size, y:y+occlusion_size] = 0
        return img