import torch
import os


# Specify the directory to save the model
model_dir = './models'
os.makedirs(model_dir, exist_ok=True)

models = [
    'cifar10_resnet20',
    'cifar100_resnet20','cifar10_resnet32',
    'cifar100_resnet32','cifar10_resnet44',
    'cifar100_resnet44','cifar10_resnet56',
    'cifar100_resnet56','cifar10_mobilenetv2_x1_4',
    'cifar100_mobilenetv2_x1_4','cifar10_vgg16_bn', 'cifar100_vgg16_bn','cifar10_vgg11_bn', 'cifar100_vgg11_bn']


for model_name in models:
    # Load the pretrained model
    model = torch.hub.load('chenyaofo/pytorch-cifar-models', model_name, pretrained=True, trust_repo=True)
    
    # Save the model's state_dict
    model_path = os.path.join(model_dir, f'{model_name}.pth')
    torch.save(model.state_dict(), model_path)
    
    print(f"Model {model_name} saved to {model_path}")