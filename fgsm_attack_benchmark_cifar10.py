import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torchattacks import FGSM
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
BATCH_SIZE  = 128
EPS         = 8/255
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
ATTACK_NAME = 'FGSM'

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
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    results = []
    for model_name in MODEL_NAMES:
        print(f"\n>>> Benchmarking {model_name} ...")
        # 1) load model + weights
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

        # 2) clean accuracy
        clean_correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(1)
                clean_correct += (preds == labels).sum().item()
                total += labels.size(0)
        clean_acc = 100 * clean_correct / total
        print(f"    Clean acc: {clean_acc:.2f}%")

        # 3) set up FGSM
        attack = FGSM(model, eps=EPS)

        # 4) run attack & measure time
        adv_correct = 0
        start_time = time.perf_counter()
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            adv_imgs = attack(imgs, labels)
            with torch.no_grad():
                adv_preds = model(adv_imgs).argmax(1)
            adv_correct += (adv_preds == labels).sum().item()
        attack_time = time.perf_counter() - start_time

        adv_acc = 100 * adv_correct / total
        print(f"    {ATTACK_NAME} acc: {adv_acc:.2f}%   time: {attack_time:.1f}s")

        # 5) record
        results.append({
            'model':         model_name,
            'attack':        ATTACK_NAME,
            'clean_acc':     round(clean_acc, 2),
            'adv_acc':       round(adv_acc, 2),
            'attack_time_s': round(attack_time, 2),
        })

    # Save results
    df = pd.DataFrame(results)
    csv_path = 'fgsm_cifar10_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nAll done — results saved to {csv_path}")

if __name__ == "__main__":
    freeze_support()
    main()
