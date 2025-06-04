"""
Complete Adversarial Attack Recreation Script
Recreates the study from "Evaluating Pretrained Deep Learning Models for Image Classification Against Individual and Ensemble Adversarial Attacks"

This script uses pre-trained distilled models from './models_distilled/' directory.
Run the defensive distillation script first to generate the distilled models.
"""

import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torchattacks import FGSM, PGD, BIM, DeepFool
from multiprocessing import freeze_support

# CONFIG - Following paper parameters exactly
MODEL_NAMES = [
    "cifar10_resnet20", "cifar10_resnet32", "cifar10_resnet44", "cifar10_resnet56",
    "cifar10_mobilenetv2_x1_4", "cifar10_vgg16_bn", "cifar10_vgg11_bn",
    "cifar100_resnet20", "cifar100_resnet32", "cifar100_resnet44", "cifar100_resnet56", 
    "cifar100_mobilenetv2_x1_4", "cifar100_vgg16_bn", "cifar100_vgg11_bn"
]

MODEL_DIR = "./models"
DATA_DIR = "./data"
RESULTS_DIR = "./results"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paper's attack parameters (ε=0.5 in pixel space = 0.5/255 in normalized space)
EPS = 0.5 / 255.0  # Paper uses ε=0.5 in pixel space
ALPHA_PGD = 1.0 / 255.0  # Step size of 1 in pixel space
ALPHA_BIM = 1.0 / 255.0  # Step size of 1 in pixel space
STEPS_PGD = 20  # Paper uses 20 iterations for PGD
STEPS_BIM = 10  # Paper uses 10 iterations for BIM

# Ensemble weights from paper
WEA_WEIGHTS = {'FGSM': 0.4, 'PGD': 0.3, 'BIM': 0.3}

os.makedirs(RESULTS_DIR, exist_ok=True)

def get_dataset_params(dataset_name):
    """Get normalization parameters for each dataset"""
    if dataset_name == 'cifar10':
        return {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010],
            'num_classes': 10,
            'dataset_class': torchvision.datasets.CIFAR10
        }
    elif dataset_name == 'cifar100':
        return {
            'mean': [0.5071, 0.4867, 0.4408],
            'std': [0.2675, 0.2565, 0.2761],
            'num_classes': 100,
            'dataset_class': torchvision.datasets.CIFAR100
        }

def get_balanced_subset(dataset, samples_per_class, num_classes):
    """Create balanced subset following paper methodology"""
    class_counts = {i: 0 for i in range(num_classes)}
    selected_indices = []

    for idx, (_, label) in enumerate(dataset):
        if class_counts[label] < samples_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
        if all(count >= samples_per_class for count in class_counts.values()):
            break

    return torch.utils.data.Subset(dataset, selected_indices)

def load_model(model_name):
    """Load pre-trained model"""
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        model_name,
        pretrained=False,
        trust_repo=True,
    ).to(DEVICE)
    
    state = torch.load(
        os.path.join(MODEL_DIR, f"{model_name}.pth"), 
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model

def create_attacks(model):
    """Create attack instances following paper parameters"""
    return {
        'FGSM': FGSM(model, eps=EPS),
        'PGD': PGD(model, eps=EPS, alpha=ALPHA_PGD, steps=STEPS_PGD),
        'BIM': BIM(model, eps=EPS, alpha=ALPHA_BIM, steps=STEPS_BIM),
        'DeepFool': DeepFool(model)
    }

def ensemble_attack_mea(attacks, images, labels):
    """Mean Ensemble Attack (MEA) - Paper equation"""
    fgsm_adv = attacks['FGSM'](images, labels)
    pgd_adv = attacks['PGD'](images, labels)
    bim_adv = attacks['BIM'](images, labels)
    
    # MEA: average the adversarial examples
    ensemble_adv = (fgsm_adv + pgd_adv + bim_adv) / 3.0
    return ensemble_adv

def ensemble_attack_wea(attacks, images, labels):
    """Weighted Ensemble Attack (WEA) - Paper equation"""
    fgsm_adv = attacks['FGSM'](images, labels)
    pgd_adv = attacks['PGD'](images, labels)
    bim_adv = attacks['BIM'](images, labels)
    
    # WEA: weighted combination
    ensemble_adv = (WEA_WEIGHTS['FGSM'] * fgsm_adv + 
                   WEA_WEIGHTS['PGD'] * pgd_adv + 
                   WEA_WEIGHTS['BIM'] * bim_adv)
    return ensemble_adv



def load_distilled_model(model_name):
    """Load pre-trained distilled model"""
    distilled_model_path = os.path.join("./models_distilled", f"{model_name}_distilled.pth")
    
    if not os.path.exists(distilled_model_path):
        print(f"Warning: Distilled model not found at {distilled_model_path}")
        return None
    
    # Load the distilled model architecture
    distilled_model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        model_name,
        pretrained=False,
        trust_repo=True,
    ).to(DEVICE)
    
    # Load the distilled weights
    state = torch.load(distilled_model_path, map_location=DEVICE)
    distilled_model.load_state_dict(state)
    distilled_model.eval()
    
    return distilled_model

def evaluate_model(model, data_loader):
    """Evaluate model accuracy"""
    correct = total = 0
    model.eval()
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100.0 * correct / total

def run_attack_evaluation(model, attack, data_loader, attack_name):
    """Run single attack and measure performance"""
    correct = total = 0
    start_time = time.time()
    
    model.eval()
    for images, labels in data_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        if attack_name in ['MEA', 'WEA']:
            attacks = create_attacks(model)
            if attack_name == 'MEA':
                adv_images = ensemble_attack_mea(attacks, images, labels)
            else:  # WEA
                adv_images = ensemble_attack_wea(attacks, images, labels)
        else:
            adv_images = attack(images, labels)
        
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    attack_time = time.time() - start_time
    accuracy = 100.0 * correct / total
    
    return accuracy, attack_time, total

def main():
    # Check if distilled models directory exists
    distilled_models_dir = "./models_distilled"
    if not os.path.exists(distilled_models_dir):
        print(f"Warning: Distilled models directory '{distilled_models_dir}' not found.")
        print("Please run the defensive distillation script first to generate distilled models.")
        print("Continuing with evaluation of original models only...")
        evaluate_distilled = False
    else:
        evaluate_distilled = True
        print(f"Found distilled models directory: {distilled_models_dir}")
    
    all_results = []
    
    for model_name in MODEL_NAMES:
        print(f"\n>>> Processing {model_name}")
        
        # Determine dataset
        dataset_name = 'cifar10' if 'cifar10' in model_name else 'cifar100'
        dataset_params = get_dataset_params(dataset_name)
        
        # Load dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_params['mean'], dataset_params['std'])
        ])
        
        test_dataset = dataset_params['dataset_class'](
            root=DATA_DIR, train=False, download=True, transform=transform
        )
        
        # Create balanced subset (150 total samples like in existing code)
        samples_per_class = 150 // dataset_params['num_classes']
        if samples_per_class == 0:
            samples_per_class = 1
        
        balanced_subset = get_balanced_subset(
            test_dataset, samples_per_class, dataset_params['num_classes']
        )
        
        data_loader = torch.utils.data.DataLoader(
            balanced_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )
        
        # Load model
        model = load_model(model_name)
        
        # Evaluate clean accuracy
        clean_acc = evaluate_model(model, data_loader)
        print(f"  Clean accuracy: {clean_acc:.2f}%")
        
        # Create attacks
        attacks = create_attacks(model)
        
        # Test individual attacks
        for attack_name, attack in attacks.items():
            print(f"  → Running {attack_name}", end="", flush=True)
            adv_acc, attack_time, num_samples = run_attack_evaluation(
                model, attack, data_loader, attack_name
            )
            accuracy_drop = clean_acc - adv_acc
            
            print(f" | Acc: {adv_acc:.2f}% | Drop: {accuracy_drop:.2f}% | Time: {attack_time:.1f}s")
            
            all_results.append({
                'model': model_name,
                'dataset': dataset_name,
                'defense': 'None',
                'attack': attack_name,
                'clean_acc': round(clean_acc, 2),
                'adv_acc': round(adv_acc, 2),
                'accuracy_drop': round(accuracy_drop, 2),
                'attack_time_s': round(attack_time, 2),
                'num_samples': num_samples
            })
        
        # Test ensemble attacks
        for ensemble_name in ['MEA', 'WEA']:
            print(f"  → Running {ensemble_name}", end="", flush=True)
            adv_acc, attack_time, num_samples = run_attack_evaluation(
                model, None, data_loader, ensemble_name
            )
            accuracy_drop = clean_acc - adv_acc
            
            print(f" | Acc: {adv_acc:.2f}% | Drop: {accuracy_drop:.2f}% | Time: {attack_time:.1f}s")
            
            all_results.append({
                'model': model_name,
                'dataset': dataset_name,
                'defense': 'None',
                'attack': ensemble_name,
                'clean_acc': round(clean_acc, 2),
                'adv_acc': round(adv_acc, 2),
                'accuracy_drop': round(accuracy_drop, 2),
                'attack_time_s': round(attack_time, 2),
                'num_samples': num_samples
            })
        
        # Evaluate distilled model if available
        if evaluate_distilled:
            # Load pre-trained distilled model
            print("  → Loading pre-trained distilled model...", end="", flush=True)
            distilled_model = load_distilled_model(model_name)
            
            if distilled_model is None:
                print(f" | Skipping {model_name} - no distilled model found")
            else:
                # Evaluate distilled model
                distilled_clean_acc = evaluate_model(distilled_model, data_loader)
                print(f" | Distilled clean acc: {distilled_clean_acc:.2f}%")
                
                # Test attacks on distilled model
                distilled_attacks = create_attacks(distilled_model)
                
                # Individual attacks on distilled model
                for attack_name, attack in distilled_attacks.items():
                    adv_acc, attack_time, num_samples = run_attack_evaluation(
                        distilled_model, attack, data_loader, attack_name
                    )
                    accuracy_drop = distilled_clean_acc - adv_acc
                    
                    all_results.append({
                        'model': model_name,
                        'dataset': dataset_name,
                        'defense': 'Distillation',
                        'attack': attack_name,
                        'clean_acc': round(distilled_clean_acc, 2),
                        'adv_acc': round(adv_acc, 2),
                        'accuracy_drop': round(accuracy_drop, 2),
                        'attack_time_s': round(attack_time, 2),
                        'num_samples': num_samples
                    })
                
                # Ensemble attacks on distilled model
                for ensemble_name in ['MEA', 'WEA']:
                    adv_acc, attack_time, num_samples = run_attack_evaluation(
                        distilled_model, None, data_loader, ensemble_name
                    )
                    accuracy_drop = distilled_clean_acc - adv_acc
                    
                    all_results.append({
                        'model': model_name,
                        'dataset': dataset_name,
                        'defense': 'Distillation',
                        'attack': ensemble_name,
                        'clean_acc': round(distilled_clean_acc, 2),
                        'adv_acc': round(adv_acc, 2),
                        'accuracy_drop': round(accuracy_drop, 2),
                        'attack_time_s': round(attack_time, 2),
                        'num_samples': num_samples
                    })
    
    # Save results
    df = pd.DataFrame(all_results)
    output_file = os.path.join(RESULTS_DIR, "complete_adversarial_evaluation.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✓ Complete results saved to {output_file}")
    
    # Print summary statistics
    print("\n=== SUMMARY ===")
    if evaluate_distilled:
        summary = df.groupby(['attack', 'defense']).agg({
            'accuracy_drop': ['mean', 'std'],
            'adv_acc': ['mean', 'std']
        }).round(2)
        print(summary)
        print(f"\nDistilled models evaluated: ✓")
    else:
        summary = df.groupby(['attack']).agg({
            'accuracy_drop': ['mean', 'std'], 
            'adv_acc': ['mean', 'std']
        }).round(2)
        print(summary)
        print(f"\nDistilled models evaluated: ✗ (run distillation script first)")

if __name__ == "__main__":
    freeze_support()
    main()