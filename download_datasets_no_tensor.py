import os
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image

def export_cifar(dataset_name='cifar10', root_dir='data', out_dir='exported'):
    """
    dataset_name: 'cifar10' or 'cifar100'
    root_dir: where the binary batches live
    out_dir: where to write your jpg/pngs
    """
    if dataset_name == 'cifar10':
        DatasetClass = CIFAR10
        num_classes = 10
    else:
        DatasetClass = CIFAR100
        num_classes = 100

    for split in ['train', 'test']:
        ds = DatasetClass(
            root=root_dir,
            train=(split=='train'),
            download=False,
            transform=None  # <-- get raw PIL images
        )

        for idx, (img, label) in enumerate(ds):
            # label is an integer 0..num_classes-1
            class_name = ds.classes[label]  # e.g. 'airplane', 'automobile', …
            save_dir = os.path.join(out_dir, dataset_name, split, f"{label}_{class_name}")
            os.makedirs(save_dir, exist_ok=True)

            # zero-pad the filename to keep ordering
            fname = f"{idx:05d}.png"
            img.save(os.path.join(save_dir, fname))

        print(f"Exported {split} split of {dataset_name} to {os.path.join(out_dir, dataset_name, split)}")

if __name__ == "__main__":
    export_cifar('cifar10', root_dir='./data', out_dir='./exported')
    export_cifar('cifar100', root_dir='./data', out_dir='./exported')
