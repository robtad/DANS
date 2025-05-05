import torchvision
import torchvision.transforms as transforms
import os

# Specify the directory to save the datasets
data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download CIFAR-10
cifar10_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
cifar10_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

# Download CIFAR-100
cifar100_train = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
cifar100_test = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

print(f"CIFAR-10 and CIFAR-100 datasets downloaded to {data_dir}")
