import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
from multiprocessing import freeze_support

# CONFIG
MODEL_DIR = './models'
DISTILLED_MODEL_DIR = './distilled_models'
DATA_DIR = './data'
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10
TEMPERATURE = 3.0  # From paper
ALPHA = 0.7  # Distillation parameter from paper
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ensure directories exist
os.makedirs(DISTILLED_MODEL_DIR, exist_ok=True)

class DistillationLoss(nn.Module):
    """
    Implements the distillation loss as described in the paper.
    Combines cross-entropy loss with distillation loss.
    """
    def __init__(self, temperature=TEMPERATURE, alpha=ALPHA):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # Standard cross-entropy loss
        ce_loss = self.criterion(student_logits, labels)
        
        # Distillation loss (KL divergence between soft targets)
        student_soft = torch.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = torch.softmax(teacher_logits / self.temperature, dim=1)
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * ce_loss
        return total_loss

def train_distilled_model(teacher_model, student_model, train_loader, val_loader, model_name, dataset_type):
    """
    Train a student model using defensive distillation
    """
    print(f"\nTraining distilled model for {model_name} on {dataset_type}")
    
    criterion = DistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)
    optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    teacher_model.eval()  # Teacher model should be in eval mode
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        student_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Get teacher predictions (soft targets)
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            
            # Get student predictions
            student_logits = student_model(images)
            
            # Calculate distillation loss
            loss = criterion(student_logits, teacher_logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        # Validation
        val_acc = evaluate_model(student_model, val_loader)
        print(f'Epoch {epoch+1}/{EPOCHS}, Train Acc: {100.*correct/total:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(student_model.state_dict())
    
    # Load best model and save
    student_model.load_state_dict(best_model_state)
    distilled_model_path = os.path.join(DISTILLED_MODEL_DIR, f'{model_name}_distilled.pth')
    torch.save(best_model_state, distilled_model_path)
    print(f'Best distilled model saved to {distilled_model_path} with val acc: {best_val_acc:.2f}%')
    
    return student_model

def evaluate_model(model, test_loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def prepare_data_loaders(dataset_type='cifar10'):
    """Prepare data loaders for CIFAR-10 or CIFAR-100"""
    if dataset_type == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        num_classes = 10
        dataset_class = torchvision.datasets.CIFAR10
    else:  # cifar100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        num_classes = 100
        dataset_class = torchvision.datasets.CIFAR100
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    train_dataset = dataset_class(root=DATA_DIR, train=True, download=True, transform=transform_train)
    test_dataset = dataset_class(root=DATA_DIR, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, num_classes

def main():
    """
    Apply defensive distillation to all models
    """
    # Models to distill
    cifar10_models = [
        "cifar10_resnet20",
        "cifar10_resnet32", 
        "cifar10_resnet44",
        "cifar10_resnet56",
        "cifar10_mobilenetv2_x1_4",
        "cifar10_vgg16_bn",
        "cifar10_vgg11_bn",
    ]
    
    cifar100_models = [
        'cifar100_resnet20',
        'cifar100_resnet32',
        'cifar100_resnet44',
        'cifar100_resnet56',
        'cifar100_mobilenetv2_x1_4',
        'cifar100_vgg16_bn',
        'cifar100_vgg11_bn',
    ]
    
    # Process CIFAR-10 models
    print("=== Processing CIFAR-10 Models ===")
    train_loader_10, test_loader_10, num_classes_10 = prepare_data_loaders('cifar10')
    
    for model_name in cifar10_models:
        print(f"\n>>> Processing {model_name}")
        
        # Load teacher model (original pretrained model)
        teacher_model = torch.hub.load(
            'chenyaofo/pytorch-cifar-models',
            model_name,
            pretrained=False,
            trust_repo=True
        ).to(DEVICE)
        teacher_state = torch.load(os.path.join(MODEL_DIR, f'{model_name}.pth'), weights_only=True)
        teacher_model.load_state_dict(teacher_state)
        
        # Create student model (same architecture, fresh weights)
        student_model = torch.hub.load(
            'chenyaofo/pytorch-cifar-models',
            model_name,
            pretrained=False,
            trust_repo=True
        ).to(DEVICE)
        
        # Train distilled model
        distilled_model = train_distilled_model(
            teacher_model, student_model, train_loader_10, test_loader_10, model_name, 'CIFAR-10'
        )
        
        # Final evaluation
        final_acc = evaluate_model(distilled_model, test_loader_10)
        print(f'Final distilled model accuracy: {final_acc:.2f}%')
    
    # Process CIFAR-100 models
    print("\n=== Processing CIFAR-100 Models ===")
    train_loader_100, test_loader_100, num_classes_100 = prepare_data_loaders('cifar100')
    
    for model_name in cifar100_models:
        print(f"\n>>> Processing {model_name}")
        
        # Load teacher model (original pretrained model)
        teacher_model = torch.hub.load(
            'chenyaofo/pytorch-cifar-models',
            model_name,
            pretrained=False,
            trust_repo=True
        ).to(DEVICE)
        teacher_state = torch.load(os.path.join(MODEL_DIR, f'{model_name}.pth'), weights_only=True)
        teacher_model.load_state_dict(teacher_state)
        
        # Create student model (same architecture, fresh weights)
        student_model = torch.hub.load(
            'chenyaofo/pytorch-cifar-models',
            model_name,
            pretrained=False,
            trust_repo=True
        ).to(DEVICE)
        
        # Train distilled model
        distilled_model = train_distilled_model(
            teacher_model, student_model, train_loader_100, test_loader_100, model_name, 'CIFAR-100'
        )
        
        # Final evaluation
        final_acc = evaluate_model(distilled_model, test_loader_100)
        print(f'Final distilled model accuracy: {final_acc:.2f}%')

if __name__ == "__main__":
    freeze_support()
    main()