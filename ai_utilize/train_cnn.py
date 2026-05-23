import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from model import SimpleCNN  # model.py에서 class 가져오기기
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # 사진 크기 조정정
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder('dataset', transform=transform) # 데이터 셋 불러오기기 및 변환환
train_size = int(0.8 * len(dataset)) 
val_size = len(dataset) - train_size 
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 디바이스 설정정
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_acc = 0.0 #최대치 이용 용용
num_epochs = 5  # 총 5번 학습시키기기
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader) 
    print(f"\n[Epoch {epoch+1}/{num_epochs}] 학습 시작!")
    for batch_idx, (images, labels) in enumerate(train_loader): # 배치 단위 학습습
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        # 실시간 진행도 출력
        progress = (batch_idx + 1) / total_batches * 100
        print(f"\r  진행률: {progress:.1f}%  |  배치: {batch_idx+1}/{total_batches}  |  현재 loss: {loss.item():.4f}", end='')
    print()  # 줄바꿈
    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, val_acc={acc:.4f}")
    # 모델 저장장
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')
        print("model saved")

print("complete")