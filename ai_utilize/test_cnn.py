import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_cnn import SimpleCNN  # train_cnn.py와 같은 폴더에 있어야 함

# 1. 데이터셋 준비
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
test_dataset = datasets.ImageFolder('_Test_Set2', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32)

# 2. 모델 불러오기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# 3. 테스트
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = correct / total
print(f"테스트셋 정확도: {acc:.4f}")