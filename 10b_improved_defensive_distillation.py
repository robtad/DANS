import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import csv

def main():
    # Local directories
    model_dir = './models'
    data_dir = './data'
    distilled_model_dir = './models_distilled'
    results_csv = './results/defensive_distillation_results.csv'

    os.makedirs(distilled_model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    temperature = 3.0
    alpha = 0.7
    num_epochs = 30

    # CIFAR-10 mean/std
    mean10 = [0.4914, 0.4822, 0.4465]
    std10 = [0.2023, 0.1994, 0.2010]

    # CIFAR-100 mean/std
    mean100 = [0.5071, 0.4867, 0.4408]
    std100 = [0.2675, 0.2565, 0.2761]

    # Define distillation loss
    def distillation_loss(student_outputs, teacher_outputs, labels, T, alpha):
        hard_loss = nn.CrossEntropyLoss()(student_outputs, labels)
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            nn.functional.log_softmax(student_outputs / T, dim=1),
            nn.functional.softmax(teacher_outputs / T, dim=1)
        ) * (T * T)
        return alpha * hard_loss + (1 - alpha) * soft_loss

    # Evaluate accuracy
    def evaluate(model, data_loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in data_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return 100 * correct / total

    # Models to distill
    models = [
        'cifar10_resnet20',
        'cifar100_resnet20',
        'cifar10_resnet32',
        'cifar100_resnet32',
        'cifar10_resnet44',
        'cifar100_resnet44',
        'cifar10_resnet56',
        'cifar100_resnet56',
        'cifar10_mobilenetv2_x1_4',
        'cifar100_mobilenetv2_x1_4',
        'cifar10_vgg16_bn',
        'cifar100_vgg16_bn',
        'cifar10_vgg11_bn',
        'cifar100_vgg11_bn'
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CSV file to store results
    with open(results_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['student_model_name', 'teacher_accuracy', 'student_accuracy', 'difference'])

        for model_name in models:
            print(f"\nDistilling {model_name}...")

            # Determine CIFAR type
            if 'cifar100' in model_name:
                num_classes = 100
                mean, std = mean100, std100
                train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True,
                                                            transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean, std)
                                                            ]))
                test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True,
                                                            transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean, std)
                                                            ]))
            else:
                num_classes = 10
                mean, std = mean10, std10
                train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                                            transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean, std)
                                                            ]))
                test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                                            transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean, std)
                                                            ]))

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

            # Load teacher model
            teacher = torch.hub.load('chenyaofo/pytorch-cifar-models', model_name, pretrained=False, trust_repo=True)
            teacher.load_state_dict(torch.load(os.path.join(model_dir, f'{model_name}.pth'), map_location=device))
            teacher.to(device)
            teacher.eval()

            # Evaluate teacher
            teacher_acc = evaluate(teacher, test_loader, device)
            print(f"Teacher accuracy: {teacher_acc:.2f}%")

            # Create student with the same architecture
            student = torch.hub.load('chenyaofo/pytorch-cifar-models', model_name, pretrained=False, trust_repo=True)
            student.to(device)

            # Optimizer
            optimizer = optim.Adam(student.parameters(), lr=learning_rate)

            # Training loop
            for epoch in range(num_epochs):
                student.train()
                running_loss = 0.0
                for imgs, labels in train_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    with torch.no_grad():
                        teacher_outputs = teacher(imgs)
                    student_outputs = student(imgs)

                    loss = distillation_loss(student_outputs, teacher_outputs, labels, temperature, alpha)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

            # Save distilled model
            distilled_model_path = os.path.join(distilled_model_dir, f'{model_name}_distilled.pth')
            torch.save(student.state_dict(), distilled_model_path)
            print(f"Distilled model saved to {distilled_model_path}")

            # Evaluate student
            student_acc = evaluate(student, test_loader, device)
            print(f"Student accuracy: {student_acc:.2f}%")

            # Difference
            diff = student_acc - teacher_acc
            print(f"Accuracy difference: {diff:.2f}%")

            # Write result to CSV
            writer.writerow([f'{model_name}_distilled', f'{teacher_acc:.2f}', f'{student_acc:.2f}', f'{diff:.2f}'])

    print("\nAll models distilled, evaluated, and results saved successfully!")



if __name__ == "__main__":
    main()