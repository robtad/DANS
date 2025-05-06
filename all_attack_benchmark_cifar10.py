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
    'cifar10_resnet20',
    'cifar10_resnet32',
    'cifar10_resnet44',
    'cifar10_resnet56',
    'cifar10_mobilenetv2_x1_4',
    'cifar10_vgg16_bn',
    'cifar10_vgg11_bn',
]
MODEL_DIR   = './models'
DATA_DIR    = './data'
BATCH_SIZE  = 64
EPS         = 0.5/255
ALPHA       = 0.01
STEPS       = 10
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SAMPLES = 128  # number of test images for both clean & adv accuracy check

# Attack constructors
ATTACK_SPECS = {
    'FGSM':      lambda m: FGSM(m, eps=EPS),
    'PGD':       lambda m: PGD(m, eps=EPS, alpha=ALPHA, steps=STEPS),
    'BIM':       lambda m: BIM(m, eps=EPS, alpha=ALPHA, steps=STEPS),
    'DeepFool':  lambda m: DeepFool(m),
}

def main():
    # Prepare CIFAR-10 test loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std =[0.2023, 0.1994, 0.2010]),
    ])
    test_ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    full_count = len(test_ds)
    print(f"Full CIFAR-10 test set size: {full_count} images")
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
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

        # Clean accuracy over MAX_SAMPLES
        clean_correct = processed_clean = 0
        for imgs, labels in test_loader:
            batch = imgs.size(0)
            if processed_clean + batch > MAX_SAMPLES:
                take = MAX_SAMPLES - processed_clean
                imgs, labels = imgs[:take], labels[:take]
                batch = take

            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                preds = model(imgs).argmax(1)
            clean_correct += (preds == labels).sum().item()
            processed_clean += batch

            if processed_clean >= MAX_SAMPLES:
                break

        clean_acc = 100 * clean_correct / processed_clean
        print(f"    Clean accuracy ({processed_clean} samples): {clean_acc:.2f}%")

        # Run each attack over the same MAX_SAMPLES
        for attack_name, make_attack in ATTACK_SPECS.items():
            print(f"  → Running {attack_name} ...", end='', flush=True)
            attack = make_attack(model)

            adv_correct = processed_adv = 0
            start = time.perf_counter()

            for imgs, labels in test_loader:
                batch = imgs.size(0)
                if processed_adv + batch > MAX_SAMPLES:
                    take = MAX_SAMPLES - processed_adv
                    imgs, labels = imgs[:take], labels[:take]
                    batch = take

                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                adv_imgs = attack(imgs, labels)
                with torch.no_grad():
                    adv_preds = model(adv_imgs).argmax(1)
                adv_correct += (adv_preds == labels).sum().item()
                processed_adv += batch

                if processed_adv >= MAX_SAMPLES:
                    break

            elapsed = time.perf_counter() - start
            adv_acc = 100 * adv_correct / processed_adv
            print(f" acc: {adv_acc:.2f}%  time: {elapsed:.1f}s  samples: {processed_adv}")

            results.append({
                'model':         model_name,
                'attack':        attack_name,
                'clean_acc':     round(clean_acc, 2),
                'adv_acc':       round(adv_acc, 2),
                'attack_time_s': round(elapsed, 2),
                'num_samples':   processed_adv,
            })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv('all_attack_benchmark_cifar10.csv', index=False)
    print("\nResults written to all_attack_benchmark_cifar10.csv")

if __name__ == "__main__":
    freeze_support()
    main()
