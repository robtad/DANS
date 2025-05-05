import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
from torchattacks import FGSM


# ─── Config ────────────────────────────────────────────────────────────────────
model_name = 'cifar10_resnet20'
model_dir  = './models'
data_dir   = './data'
eps        = 8/255            # common ε for CIFAR (in [0,1] scale)
batch_size = 128
device     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── Load model & weights ───────────────────────────────────────────────────────
model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    model_name,
    pretrained=False,
    trust_repo=True
).to(device)
state = torch.load(
    os.path.join(model_dir, f'{model_name}.pth'),
    weights_only=True
)
model.load_state_dict(state)
model.eval()

# ─── DataLoader ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.4914, 0.4822, 0.4465],
      std =[0.2023, 0.1994, 0.2010]
    ),
])
test_ds = torchvision.datasets.CIFAR10(
    root=data_dir, train=False, download=False, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=batch_size, shuffle=False
)

# ─── FGSM function ───────────────────────────────────────────────────────────────
# def fgsm_attack(images, gradients, epsilon):
#     # images: normalized inputs; gradients: ∂loss/∂images
#     sign_grad = gradients.sign()
#     adv_images = images + epsilon * sign_grad
#     # clamp to [min,max] in normalized space:
#     # for each channel: (0 - mean)/std  →  (1 - mean)/std
#     clamp_min = [(0.0 - m)/s for m, s in zip([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010])]
#     clamp_max = [(1.0 - m)/s for m, s in zip([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010])]
#     for c in range(3):
#         adv_images[:,c] = torch.clamp(adv_images[:,c], clamp_min[c], clamp_max[c])
#     return adv_images


# ─── Create the FGSM attack from Torchattacks ──────────────────────────────────
attack = FGSM(model, eps=eps)

# ─── Run attack & eval ───────────────────────────────────────────────────────────
clean_correct = 0
adv_correct   = 0
total         = 0

for imgs, labels in test_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    total += labels.size(0)

    # —— clean accuracy —— 
    with torch.no_grad():
        preds = model(imgs).argmax(1)
    clean_correct += (preds == labels).sum().item()

    # —— generate adversarial examples —— 
    imgs.requires_grad_(True)
    outputs = model(imgs)
    loss    = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    grads = imgs.grad.data

    # adv_imgs = fgsm_attack(imgs, grads, eps)
    adv_imgs = attack(imgs, labels)
    

    # —— adversarial accuracy —— 
    with torch.no_grad():
        adv_preds = model(adv_imgs).argmax(1)
    adv_correct += (adv_preds == labels).sum().item()

# ─── Report ─────────────────────────────────────────────────────────────────────
clean_acc = 100 * clean_correct / total
adv_acc   = 100 * adv_correct   / total

print(f"Model: {model_name}")
print(f"  Clean accuracy: {clean_acc:5.2f}%")
print(f"   FGSM (ε={eps:.3f}) acc: {adv_acc:5.2f}%")
