import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torchattacks import FGSM, PGD, BIM, DeepFool
from multiprocessing import freeze_support

# CONFIG
MODEL_NAMES = [
    'cifar100_resnet20',
    'cifar100_resnet32',
    'cifar100_resnet44',
    'cifar100_resnet56',
    'cifar100_mobilenetv2_x1_4',
    'cifar100_vgg16_bn',
    'cifar100_vgg11_bn',
]
MODEL_DIR = './models'
DATA_DIR = './data'
BATCH_SIZE = 64
EPS = 0.5 / 255  # paper’s 0.5 in pixel-space → 0.5/255 here
ALPHA = 0.01
STEPS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SAMPLES = 500  # number of test images for both clean & adv

# Attack constructors
ATTACK_SPECS = {
    'FGSM': lambda m: FGSM(m, eps=EPS),
    'PGD': lambda m: PGD(m, eps=EPS, alpha=ALPHA, steps=STEPS),
    'BIM': lambda m: BIM(m, eps=EPS, alpha=ALPHA, steps=STEPS),
    'DeepFool': lambda m: DeepFool(m),
}

def get_balanced_subset(dataset, samples_per_class):
    """
    Builds a balanced subset of the dataset with samples_per_class images per class.
    """
    num_classes = 100
    class_counts = {i: 0 for i in range(num_classes)}
    selected_indices = []

    for idx, (_, label) in enumerate(dataset):
        if class_counts[label] < samples_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
        if all(count >= samples_per_class for count in class_counts.values()):
            break

    balanced_subset = torch.utils.data.Subset(dataset, selected_indices)
    return balanced_subset

def main():
    # Prepare CIFAR-100 test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])
    test_ds = torchvision.datasets.CIFAR100(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    full_count = len(test_ds)  # 10,000
    print(f"Full CIFAR-100 test set size: {full_count} images")

    # Calculate samples per class
    num_classes = 100
    samples_per_class = max(1, MAX_SAMPLES // num_classes)
    total_samples = samples_per_class * num_classes
    print(f"Using {samples_per_class} samples per class → {total_samples} total samples")

    # Build balanced subset
    balanced_subset = get_balanced_subset(test_ds, samples_per_class)
    test_loader = torch.utils.data.DataLoader(
        balanced_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    results = []

    for model_name in MODEL_NAMES:
        print(f"\n>>> Loading model {model_name}")
        model = torch.hub.load(
            'chenyaofo/pytorch-cifar-models',
            model_name,
            pretrained=False,
            trust_repo=True
        ).to(DEVICE)
        state = torch.load(
            os.path.join(MODEL_DIR, f'{model_name}.pth'),
            weights_only=True
        )
        model.load_state_dict(state)
        model.eval()

        # Clean accuracy
        clean_correct = processed_clean = 0
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                preds = model(imgs).argmax(1)
            clean_correct += (preds == labels).sum().item()
            processed_clean += imgs.size(0)

        clean_acc = 100 * clean_correct / processed_clean
        print(f"    Clean accuracy ({processed_clean} samples): {clean_acc:.2f}%")

        # Adversarial accuracy
        for attack_name, make_attack in ATTACK_SPECS.items():
            print(f"  → Running {attack_name} ...", end='', flush=True)
            attack = make_attack(model)

            adv_correct = processed_adv = 0
            start = time.perf_counter()

            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                adv_imgs = attack(imgs, labels)
                with torch.no_grad():
                    adv_preds = model(adv_imgs).argmax(1)
                adv_correct += (adv_preds == labels).sum().item()
                processed_adv += imgs.size(0)

            elapsed = time.perf_counter() - start
            adv_acc = 100 * adv_correct / processed_adv
            print(f" acc: {adv_acc:.2f}%  time: {elapsed:.1f}s  samples: {processed_adv}")

            results.append({
                'model': model_name,
                'attack': attack_name,
                'clean_acc': round(clean_acc, 2),
                'adv_acc': round(adv_acc, 2),
                'attack_time_s': round(elapsed, 2),
                'num_samples': processed_adv,
            })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv('./results/all_attack_benchmark_cifar100_balanced.csv', index=False)
    print("\nResults written to all_attack_benchmark_cifar100_balanced.csv")

if __name__ == "__main__":
    freeze_support()
    main()
