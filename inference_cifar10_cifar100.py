import torch
import torchvision
import torchvision.transforms as transforms
import os

# Config 
model_dir = './models'
data_dir  = './data'
batch_size = 128

# CIFAR-10 mean/std
mean10 = [0.4914, 0.4822, 0.4465]
std10  = [0.2023, 0.1994, 0.2010]

# CIFAR-100 mean/std
mean100 = [0.5071, 0.4867, 0.4408]
std100  = [0.2675, 0.2565, 0.2761]

# Prepare datasets 
transform10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean10, std=std10)
])
transform100 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean100, std=std100)
])

test10 = torchvision.datasets.CIFAR10(
    root=data_dir, train=False, download=True, transform=transform10
)
test100 = torchvision.datasets.CIFAR100(
    root=data_dir, train=False, download=True, transform=transform100
)

loader10 = torch.utils.data.DataLoader(test10, batch_size=batch_size, shuffle=False)
loader100 = torch.utils.data.DataLoader(test100, batch_size=batch_size, shuffle=False)

# Function to evaluate one model 
def evaluate(model_name, dataset_loader, cifar_type='10'):
    # load architecture
    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        model_name,
        pretrained=False,
        trust_repo=True
    )
    # load saved weights only
    state = torch.load(
        os.path.join(model_dir, f'{model_name}.pth'),
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for imgs, labels in dataset_loader:
            outputs = model(imgs)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"{model_name:<25} CIFAR-{cifar_type}  Accuracy: {acc:5.2f}%")

# Run evaluations 

models = [
    'cifar10_resnet20',
    'cifar100_resnet20','cifar10_resnet32',
    'cifar100_resnet32','cifar10_resnet44',
    'cifar100_resnet44','cifar10_resnet56',
    'cifar100_resnet56','cifar10_mobilenetv2_x1_4',
    'cifar100_mobilenetv2_x1_4','cifar10_vgg16_bn', 'cifar100_vgg16_bn','cifar10_vgg11_bn', 'cifar100_vgg11_bn']



print("\nCIFAR-10 models:")
for m in models:
    if m.startswith('cifar10_'):
        evaluate(m, loader10, cifar_type='10')

print("\nCIFAR-100 models:")
for m in models:
    if m.startswith('cifar100_'):
        evaluate(m, loader100, cifar_type='100')
