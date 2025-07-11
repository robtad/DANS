import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
import time
from torchattacks import FGSM, PGD


def get_balanced_small_subset(dataset, samples_per_class=10, num_classes=10):
    """Get a small balanced subset for fast testing"""
    class_counts = {i: 0 for i in range(num_classes)}
    selected_indices = []

    for idx, (_, label) in enumerate(dataset):
        if class_counts[label] < samples_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
        if all(count >= samples_per_class for count in class_counts.values()):
            break

    return torch.utils.data.Subset(dataset, selected_indices)


def custom_bim_attack(model, images, labels, eps=0.5 / 255, alpha=2.0 / 255, steps=10):
    """Custom Basic Iterative Method (BIM) attack implementation"""
    model.eval()

    # Clone images to avoid modifying original
    images = images.clone().detach().to(images.device)
    labels = labels.clone().detach().to(labels.device)

    # Initialize adversarial images
    adv_images = images.clone().detach()

    for step in range(steps):
        adv_images.requires_grad = True

        # Forward pass
        outputs = model(adv_images)

        # Calculate loss
        loss = F.cross_entropy(outputs, labels)

        # Backward pass
        loss.backward()

        # Get gradients
        grad = adv_images.grad.data

        # Apply FGSM step
        adv_images = adv_images.detach() + alpha * torch.sign(grad)

        # Clip perturbation to be within eps of original image
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images


def ensemble_attack_mea(model, images, labels, device, eps=0.5 / 255, alpha=2.0 / 255):
    """Mean Ensemble Attack - averages perturbations from FGSM, PGD, BIM"""
    model.eval()

    # Create individual attacks
    fgsm = FGSM(model, eps=eps)
    pgd = PGD(model, eps=eps, alpha=alpha, steps=20)

    # Generate adversarial examples
    fgsm_adv = fgsm(images, labels)
    pgd_adv = pgd(images, labels)
    bim_adv = custom_bim_attack(model, images, labels, eps=eps, alpha=alpha, steps=10)

    # Calculate perturbations
    fgsm_pert = fgsm_adv - images
    pgd_pert = pgd_adv - images
    bim_pert = bim_adv - images

    # Mean ensemble: average the perturbations
    mean_pert = (fgsm_pert + pgd_pert + bim_pert) / 3

    # Apply averaged perturbation
    ensemble_adv = images + mean_pert

    # Clamp to valid range
    ensemble_adv = torch.clamp(ensemble_adv, 0, 1)

    return ensemble_adv


def ensemble_attack_wea(model, images, labels, device, eps=0.5 / 255, alpha=2.0 / 255):
    """Weighted Ensemble Attack - FGSM(0.4), PGD(0.3), BIM(0.3)"""
    model.eval()

    # Create individual attacks
    fgsm = FGSM(model, eps=eps)
    pgd = PGD(model, eps=eps, alpha=alpha, steps=20)

    # Generate adversarial examples
    fgsm_adv = fgsm(images, labels)
    pgd_adv = pgd(images, labels)
    bim_adv = custom_bim_attack(model, images, labels, eps=eps, alpha=alpha, steps=10)

    # Weighted ensemble as per paper: FGSM(0.4), PGD(0.3), BIM(0.3)
    ensemble_adv = 0.4 * fgsm_adv + 0.3 * pgd_adv + 0.3 * bim_adv

    # Clamp to valid range
    ensemble_adv = torch.clamp(ensemble_adv, 0, 1)

    return ensemble_adv


def evaluate_model(model, data_loader, device):
    """Evaluate clean accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def evaluate_attack(
    model, data_loader, attack, device, attack_name, eps=0.5 / 255, alpha=2.0 / 255
):
    """Evaluate adversarial accuracy with timing"""
    model.eval()
    correct = 0
    total = 0

    start_time = time.time()

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        if attack_name == "MEA":
            adv_images = ensemble_attack_mea(model, images, labels, device, eps, alpha)
        elif attack_name == "WEA":
            adv_images = ensemble_attack_wea(model, images, labels, device, eps, alpha)
        elif attack_name == "BIM":
            adv_images = custom_bim_attack(model, images, labels, eps, alpha, steps=10)
        else:
            adv_images = attack(images, labels)

        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    elapsed_time = time.time() - start_time
    return 100 * correct / total, elapsed_time


def get_dataset_info(dataset_name):
    """Get dataset-specific information"""
    if dataset_name == "cifar10":
        return {
            "num_classes": 10,
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2023, 0.1994, 0.2010],
            "dataset_class": torchvision.datasets.CIFAR10,
        }
    else:  # cifar100
        return {
            "num_classes": 100,
            "mean": [0.5071, 0.4867, 0.4408],
            "std": [0.2675, 0.2565, 0.2761],
            "dataset_class": torchvision.datasets.CIFAR100,
        }


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "./models"
    distilled_model_dir = "./models_distilled"
    data_dir = "./data"
    results_dir = "./results"

    print(f"Using device: {device}")

    # All models to test
    models = [
        "cifar10_resnet20",
        "cifar10_resnet32",
        "cifar10_resnet44",
        "cifar10_resnet56",
        "cifar10_mobilenetv2_x1_4",
        "cifar10_vgg16_bn",
        "cifar10_vgg11_bn",
        "cifar100_resnet20",
        "cifar100_resnet32",
        "cifar100_resnet44",
        "cifar100_resnet56",
        "cifar100_mobilenetv2_x1_4",
        "cifar100_vgg16_bn",
        "cifar100_vgg11_bn",
    ]

    # Paper parameters
    eps = 0.5 / 255  # 0.5 in paper, normalized for [0,1] range
    alpha = 2.0 / 255  # Step size

    print(f"Parameters: eps={eps:.6f}, alpha={alpha:.6f}")

    # Define attacks
    attacks = {
        "FGSM": lambda m: FGSM(m, eps=eps),
        "PGD": lambda m: PGD(m, eps=eps, alpha=alpha, steps=20),
        "BIM": None,  # Custom implementation
        "MEA": None,  # Custom ensemble attack
        "WEA": None,  # Custom ensemble attack
    }

    all_results = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"{'='*60}")

        # Determine dataset type
        dataset_name = "cifar10" if "cifar10" in model_name else "cifar100"
        dataset_info = get_dataset_info(dataset_name)

        # Prepare dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=dataset_info["mean"], std=dataset_info["std"]
                ),
            ]
        )

        test_dataset = dataset_info["dataset_class"](
            root=data_dir, train=False, download=True, transform=transform
        )

        # Use 10 samples per class
        samples_per_class = 10
        test_subset = get_balanced_small_subset(
            test_dataset, samples_per_class, dataset_info["num_classes"]
        )
        test_loader = torch.utils.data.DataLoader(
            test_subset, batch_size=32, shuffle=False, num_workers=0
        )

        print(f"Dataset: {dataset_name.upper()}")
        print(f"Using {len(test_subset)} images ({samples_per_class} per class)")

        # Test both original and distilled models
        for model_type in ["original", "distilled"]:
            model_path = os.path.join(
                model_dir if model_type == "original" else distilled_model_dir,
                (
                    f"{model_name}.pth"
                    if model_type == "original"
                    else f"{model_name}_distilled.pth"
                ),
            )

            if not os.path.exists(model_path):
                print(f" {model_type.capitalize()} model not found: {model_path}")
                continue

            print(f"\n--- {model_type.capitalize()} Model ---")

            # Load model
            model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models",
                model_name,
                pretrained=False,
                trust_repo=True,
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            # Evaluate clean accuracy
            clean_acc = evaluate_model(model, test_loader, device)
            print(f"Clean accuracy: {clean_acc:.2f}%")

            # Test each attack
            for attack_name, attack_constructor in attacks.items():
                print(f"  {attack_name}:", end=" ")

                if attack_constructor is not None:
                    attack = attack_constructor(model)
                    adv_acc, attack_time = evaluate_attack(
                        model, test_loader, attack, device, attack_name, eps, alpha
                    )
                else:
                    # Custom attacks (BIM, MEA, WEA)
                    adv_acc, attack_time = evaluate_attack(
                        model, test_loader, None, device, attack_name, eps, alpha
                    )

                # Calculate accuracy drop according to paper formula
                dropped_percentage = ((clean_acc - adv_acc) / clean_acc) * 100

                print(
                    f"{clean_acc:.1f}% → {adv_acc:.1f}% (Dropped: {dropped_percentage:.1f}%) [{attack_time:.1f}s]"
                )

                all_results.append(
                    {
                        "model_name": model_name,
                        "dataset": dataset_name.upper(),
                        "model_type": model_type.capitalize(),
                        "attack": attack_name,
                        "clean_acc": round(clean_acc, 2),
                        "adv_acc": round(adv_acc, 2),
                        "dropped_percentage": round(dropped_percentage, 1),
                        "attack_time": round(attack_time, 2),
                        "num_samples": len(test_subset),
                    }
                )

    # Save comprehensive results
    df = pd.DataFrame(all_results)
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "comprehensive_attack_evaluation.csv")
    df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*60}")

    # Print summary statistics
    print(f"\nModels tested: {len(models)}")
    print(f"Total evaluations: {len(all_results)}")
    print(f"Results saved to: {output_file}")

    # Show sample results
    print(f"\nSample Results (first 10 rows):")
    print(df.head(10).to_string(index=False))

    print(f"\nComprehensive evaluation completed!")


if __name__ == "__main__":
    main()
