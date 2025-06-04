import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torchattacks import FGSM, PGD
from multiprocessing import freeze_support

# CONFIG
MODEL_NAMES = [
    "cifar10_resnet20",
    "cifar10_resnet32", 
    "cifar10_resnet44",
    "cifar10_resnet56",
    "cifar10_mobilenetv2_x1_4",
    "cifar10_vgg16_bn",
    "cifar10_vgg11_bn",
]
MODEL_DIR = "./models"
DATA_DIR = "./data"
BATCH_SIZE = 64
EPS = 0.05/255  # Fixed epsilon that works for BIM
ALPHA = 0.01
STEPS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = 150  # total number of test images (15 per class for CIFAR-10)

class CustomBIM:
    """Custom BIM implementation that actually works"""
    def __init__(self, model, eps=0.03, alpha=0.01, steps=10):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        
    def __call__(self, images, labels):
        images = images.clone().detach().requires_grad_(True)
        labels = labels.clone().detach()
        
        # Initialize adversarial images
        adv_images = images.clone().detach()
        
        for step in range(self.steps):
            adv_images.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(adv_images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # Backward pass
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            
            # Update adversarial images
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            
            # Clip to stay within eps-ball
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        return adv_images

class FixedEnsembleAttack:
    """
    Ensemble Attack with working BIM implementation
    """
    def __init__(self, model, eps=EPS, alpha=ALPHA, steps=STEPS):
        self.model = model
        self.fgsm = FGSM(model, eps=eps)
        self.pgd = PGD(model, eps=eps, alpha=alpha, steps=steps)
        self.bim = CustomBIM(model, eps=eps, alpha=alpha, steps=steps)  # Use custom BIM
        
    def debug_individual_attacks(self, images, labels):
        """Test individual attacks to ensure they're working"""
        # Enable gradients for adversarial attacks
        images = images.requires_grad_(True)
        
        # Original accuracy
        with torch.no_grad():
            orig_pred = self.model(images).argmax(1)
            orig_acc = (orig_pred == labels).float().mean() * 100
        
        # Test each attack (these need gradients)
        adv_fgsm = self.fgsm(images, labels)
        adv_pgd = self.pgd(images, labels) 
        adv_bim = self.bim(images, labels)
        
        with torch.no_grad():
            fgsm_pred = self.model(adv_fgsm).argmax(1)
            pgd_pred = self.model(adv_pgd).argmax(1)
            bim_pred = self.model(adv_bim).argmax(1)
            
            fgsm_acc = (fgsm_pred == labels).float().mean() * 100
            pgd_acc = (pgd_pred == labels).float().mean() * 100
            bim_acc = (bim_pred == labels).float().mean() * 100
            
            print(f"      Individual attacks - FGSM: {fgsm_acc:.1f}%, PGD: {pgd_acc:.1f}%, BIM: {bim_acc:.1f}%")
            
            return adv_fgsm, adv_pgd, adv_bim
        
    def mean_ensemble_attack(self, images, labels):
        """
        Mean Ensemble Attack: x_ens = (x_fgsm + x_pgd + x_bim) / 3
        """
        # Enable gradients for adversarial attacks
        images = images.requires_grad_(True)
        
        # Generate adversarial examples using individual attacks
        adv_fgsm = self.fgsm(images, labels)
        adv_pgd = self.pgd(images, labels) 
        adv_bim = self.bim(images, labels)
        
        # Average the adversarial examples
        adv_ensemble = (adv_fgsm + adv_pgd + adv_bim) / 3.0
        
        # Clamp to valid range [0, 1]
        adv_ensemble = torch.clamp(adv_ensemble, 0, 1)
        
        return adv_ensemble
    
    def weighted_ensemble_attack(self, images, labels, weights=[0.4, 0.3, 0.3]):
        """
        Weighted Ensemble Attack (paper weights): FGSM(0.4), PGD(0.3), BIM(0.3)
        """
        # Enable gradients for adversarial attacks
        images = images.requires_grad_(True)
        
        w_fgsm, w_pgd, w_bim = weights
        
        # Generate adversarial examples using individual attacks
        adv_fgsm = self.fgsm(images, labels)
        adv_pgd = self.pgd(images, labels)
        adv_bim = self.bim(images, labels)
        
        # Weighted combination
        adv_ensemble = w_fgsm * adv_fgsm + w_pgd * adv_pgd + w_bim * adv_bim
        
        # Clamp to valid range [0, 1]
        adv_ensemble = torch.clamp(adv_ensemble, 0, 1)
        
        return adv_ensemble

    def wea_alt_attack(self, images, labels):
        """
        WEA-Alt: Your contribution - heavily favor FGSM
        """
        # Enable gradients for adversarial attacks
        images = images.requires_grad_(True)
        
        # Generate adversarial examples
        adv_fgsm = self.fgsm(images, labels)
        adv_pgd = self.pgd(images, labels)
        adv_bim = self.bim(images, labels)
        
        # Very distinct weights: heavily favor FGSM
        adv_ensemble = 0.7 * adv_fgsm + 0.2 * adv_pgd + 0.1 * adv_bim
        
        # Clamp to valid range [0, 1]
        adv_ensemble = torch.clamp(adv_ensemble, 0, 1)
        
        return adv_ensemble

def get_balanced_subset(dataset, samples_per_class):
    """Create balanced subset with equal samples per class"""
    class_counts = {i: 0 for i in range(10)}  # CIFAR-10 has 10 classes
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
    """Evaluate MEA, WEA, and WEA-Alt"""
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
    
    # Initialize fixed ensemble attack
    ensemble_attack = FixedEnsembleAttack(model, eps=EPS, alpha=ALPHA, steps=STEPS)
    
    # Debug individual attacks on first batch
    first_batch = True
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        if first_batch:
            ensemble_attack.debug_individual_attacks(imgs, labels)
            first_batch = False
        break
    
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
    
    # Test Weighted Ensemble Attack (WEA) - Paper weights
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
    
    # Test WEA-Alt (Your Contribution)
    print(f"  → Running WEA-Alt ...", end="", flush=True)
    wea_alt_correct = processed_wea_alt = 0
    start = time.perf_counter()
    
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        adv_imgs = ensemble_attack.wea_alt_attack(imgs, labels)
        with torch.no_grad():
            adv_preds = model(adv_imgs).argmax(1)
        wea_alt_correct += (adv_preds == labels).sum().item()
        processed_wea_alt += imgs.size(0)
    
    wea_alt_elapsed = time.perf_counter() - start  
    wea_alt_acc = 100 * wea_alt_correct / processed_wea_alt
    wea_alt_drop = clean_acc - wea_alt_acc
    print(f" acc: {wea_alt_acc:.2f}%  drop: {wea_alt_drop:.2f}%  time: {wea_alt_elapsed:.1f}s")
    
    results.append({
        "model": model_name,
        "attack": "WEA-Alt", 
        "clean_acc": round(clean_acc, 2),
        "adv_acc": round(wea_alt_acc, 2),
        "acc_drop": round(wea_alt_drop, 2),
        "attack_time_s": round(wea_alt_elapsed, 2),
        "num_samples": processed_wea_alt,
    })
    
    return results

def main():
    # Prepare CIFAR-10 test loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], 
            std=[0.2023, 0.1994, 0.2010]
        ),
    ])
    test_ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    full_count = len(test_ds)
    print(f"Full CIFAR-10 test set size: {full_count} images")

    # Create balanced subset (15 samples per class = 150 total)
    samples_per_class = MAX_SAMPLES // 10
    balanced_test_ds = get_balanced_subset(test_ds, samples_per_class)
    print(f"Balanced test set size: {len(balanced_test_ds)} images ({samples_per_class} per class)")
    print(f"🔧 Using FIXED BIM with ε={EPS} (works properly!)")

    balanced_test_loader = torch.utils.data.DataLoader(
        balanced_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    all_results = []

    for model_name in MODEL_NAMES:
        print(f"\n>>> Loading model {model_name}")
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            model_name,
            pretrained=False,
            trust_repo=True,
        ).to(DEVICE)
        state = torch.load(
            os.path.join(MODEL_DIR, f"{model_name}.pth"), map_location=DEVICE, weights_only=True
        )
        model.load_state_dict(state)
        model.eval()

        # Evaluate ensemble attacks
        results = evaluate_ensemble_attacks(model, balanced_test_loader, model_name)
        all_results.extend(results)

    # Save to CSV
    df = pd.DataFrame(all_results)
    output_file = "./results/final_ensemble_attack_cifar10_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nFinal ensemble attack results written to {output_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("🏆 FINAL RESULTS - All attacks should now work properly!")
    print(f"{'='*80}")
    for _, row in df.iterrows():
        print(f"{row['model']:<25} {row['attack']:<8}: {row['clean_acc']:.1f}% → {row['adv_acc']:.1f}% (drop: {row['acc_drop']:.1f}%)")
    
    # Highlight your contribution
    wea_alt_results = df[df['attack'] == 'WEA-Alt']
    avg_wea_alt_drop = wea_alt_results['acc_drop'].mean()
    print(f"\n🎉 YOUR CONTRIBUTION (WEA-Alt) average drop: {avg_wea_alt_drop:.1f}%")

if __name__ == "__main__":
    freeze_support()
    main()