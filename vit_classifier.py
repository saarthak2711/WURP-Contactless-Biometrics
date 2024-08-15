import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='train', transform=transform)
test_dataset = datasets.ImageFolder(root='test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTClassifier, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        
        self.activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook
        
        
        self.model.blocks[-1].attn.register_forward_hook(get_activation('last_attention'))
        self.model.blocks[-1].mlp.register_forward_hook(get_activation('last_mlp'))
        self.model.blocks[-1].norm1.register_forward_hook(get_activation('last_norm1'))
        
        
    def forward(self, x):
        return self.model(x)


num_classes = 336
vit_model = ViTClassifier(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit_model.parameters(), lr=0.001)


print("Training started...")
for epoch in range(10):
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{10}', leave=False)
    
    for inputs, labels in progress_bar:
        optimizer.zero_grad()
        
        outputs = vit_model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({'loss': running_loss / total, 'accuracy': 100 * correct / total})

print('Training finished.')


vit_model.eval()
top_images = []
top_activations = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = vit_model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=30, dim=1)
        
        for i in range(inputs.size(0)):
            if labels[i] in top_indices[i]:
                top_images.append(inputs[i])
                top_activations.append({
                    'last_attention': vit_model.activation['last_attention'][i],
                    'last_mlp': vit_model.activation['last_mlp'][i],
                    'last_norm1': vit_model.activation['last_norm1'][i],
                    
                })
                if len(top_images) == 30:
                    break
        if len(top_images) == 30:
            break


for i in range(len(top_images)):
    plt.figure(figsize=(15, 7))
    
    
    plt.subplot(1, 4, 1)
    plt.imshow(transforms.functional.to_pil_image(top_images[i]))
    plt.title(f'Top Image {i + 1}')
    plt.axis('off')
    
    
    plt.subplot(1, 4, 2)
    plt.imshow(top_activations[i]['last_attention'][0].cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.title(f'Last Attention Map')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(top_activations[i]['last_mlp'][0].cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.title(f'Last MLP Activation')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(top_activations[i]['last_norm1'][0].cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.title(f'Last Norm1 Activation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = vit_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Test Accuracy: {:.2f}%'.format(accuracy))
