import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torchattacks import FGSM, PGD, BIM
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
EPS = 0.5 / 255  # Paper uses ε = 0.5 in pixel space
ALPHA = 0.01
STEPS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SAMPLES = 1500  # number of test images (15 per class for CIFAR-100)

class EnsembleAttack:
    """
    Implements Mean Ensemble Attack (MEA) and Weighted Ensemble Attack (WEA)
    as described in the paper.
    """
    def __init__(self, model, eps=EPS, alpha=ALPHA, steps=STEPS):
        self.model = model
        self.fgsm = FGSM(model, eps=eps)
        self.pgd = PGD(model, eps=eps, alpha=alpha, steps=steps)
        self.bim = BIM(model, eps=eps, alpha=alpha, steps=steps)
        
    def mean_ensemble_attack(self, images, labels):
        """
        Mean Ensemble Attack: x_ens = (x_fgsm + x_pgd + x_bim) / 3
        """
        # Generate adversarial examples using individual attacks
        adv_fgsm = self.fgsm(images, labels)
        adv_pgd = self.pgd(images, labels) 
        adv_bim = self.bim(images, labels)
        
        # Average the adversarial examples
        adv_ensemble = (adv_fgsm + adv_pgd + adv_bim) / 3.0
        
        # Clamp to valid range [0, 1] after normalization
        adv_ensemble = torch.clamp(adv_ensemble, 0, 1)
        
        return adv_ensemble
    
    def weighted_ensemble_attack(self, images, labels, weights=[0.4, 0.3, 0.3]):
        """
        Weighted Ensemble Attack: x_wens = w_fgsm * x_fgsm + w_pgd * x_pgd + w_bim * x_bim
        Default weights from paper: FGSM(0.4), PGD(0.3), BIM(0.3)
        """
        w_fgsm, w_pgd, w_bim = weights
        
        # Generate adversarial examples using individual attacks
        adv_fgsm = self.fgsm(images, labels)
        adv_pgd = self.pgd(images, labels)
        adv_bim = self.bim(images, labels)
        
        # Weighted combination
        adv_ensemble = w_fgsm * adv_fgsm + w_pgd * adv_pgd + w_bim * adv_bim
        
        # Clamp to valid range [0, 1] after normalization  
        adv_ensemble = torch.clamp(adv_ensemble, 0, 1)
        
        return adv_ensemble

def get_balanced_subset(dataset, samples_per_class):
    """Create balanced subset with equal samples per class"""
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

def evaluate_ensemble_attacks(model, test_loader, model_name):
    """Evaluate both MEA and WEA on the given model"""
    results = []
    
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
    
    # Initialize ensemble attack
    ensemble_attack = EnsembleAttack(model, eps=EPS, alpha=ALPHA, steps=STEPS)
    
    # Test Mean Ensemble Attack (MEA)
    print(f"  → Running MEA ...", end="", flush=True)
    mea_correct = processed_mea = 0
    start = time.perf_counter()
    
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        adv_imgs = ensemble_attack.mean_ensemble_attack(imgs, labels)
        with torch.no_grad():
            adv_preds = model(adv_imgs).argmax(1)
        mea_correct += (adv_preds == labels).sum().item()
        processed_mea += imgs.size(0)
    
    mea_elapsed = time.perf_counter() - start
    mea_acc = 100 * mea_correct / processed_mea
    mea_drop = clean_acc - mea_acc
    print(f" acc: {mea_acc:.2f}%  drop: {mea_drop:.2f}%  time: {mea_elapsed:.1f}s")
    
    results.append({
        "model": model_name,
        "attack": "MEA",
        "clean_acc": round(clean_acc, 2),
        "adv_acc": round(mea_acc, 2),
        "acc_drop": round(mea_drop, 2),
        "attack_time_s": round(mea_elapsed, 2),
        "num_samples": processed_mea,
    })
    
    # Test Weighted Ensemble Attack (WEA)  
    print(f"  → Running WEA ...", end="", flush=True)
    wea_correct = processed_wea = 0
    start = time.perf_counter()
    
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        adv_imgs = ensemble_attack.weighted_ensemble_attack(imgs, labels, weights=[0.4, 0.3, 0.3])
        with torch.no_grad():
            adv_preds = model(adv_imgs).argmax(1)
        wea_correct += (adv_preds == labels).sum().item()
        processed_wea += imgs.size(0)
    
    wea_elapsed = time.perf_counter() - start  
    wea_acc = 100 * wea_correct / processed_wea
    wea_drop = clean_acc - wea_acc
    print(f" acc: {wea_acc:.2f}%  drop: {wea_drop:.2f}%  time: {wea_elapsed:.1f}s")
    
    results.append({
        "model": model_name,
        "attack": "WEA", 
        "clean_acc": round(clean_acc, 2),
        "adv_acc": round(wea_acc, 2),
        "acc_drop": round(wea_drop, 2),
        "attack_time_s": round(wea_elapsed, 2),
        "num_samples": processed_wea,
    })
    
    return results

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

    all_results = []

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

        # Evaluate ensemble attacks
        results = evaluate_ensemble_attacks(model, test_loader, model_name)
        all_results.extend(results)

    # Save to CSV
    df = pd.DataFrame(all_results)
    output_file = './results/ensemble_attack_cifar100_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nEnsemble attack results written to {output_file}")

if __name__ == "__main__":
    freeze_support()
    main()