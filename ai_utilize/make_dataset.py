import os
import yaml
import shutil
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN  # 모델 정의는 아래 참고

# 경로 설정
yml_dir = "Annotations_RGB/Annotations_RGB"
img_dir = "Images_RGB/Images_RGB"
out_dir = "dataset"
os.makedirs(f"{out_dir}/wheelchair", exist_ok=True)
os.makedirs(f"{out_dir}/non_wheelchair", exist_ok=True)

for yml_file in os.listdir(yml_dir):
    if not yml_file.endswith(".yml"):
        continue
    with open(os.path.join(yml_dir, yml_file), "r") as f:
        data = yaml.safe_load(f)
    annotation = data["annotation"]
    # object가 없으면 빈 리스트로 처리
    objects = annotation.get("object", [])
    # object가 단일 객체일 때도 리스트로 변환
    if isinstance(objects, dict):
        objects = [objects]
    has_wheelchair = any(obj["name"] == "wheelchair" for obj in objects)
    img_name = annotation["filename"]
    src_img = os.path.join(img_dir, img_name)
    if os.path.exists(src_img):
        dst_dir = f"{out_dir}/wheelchair" if has_wheelchair else f"{out_dir}/non_wheelchair"
        shutil.copy(src_img, os.path.join(dst_dir, img_name))

# 데이터셋 준비
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder('dataset', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size]) 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 모델, 손실함수, 옵티마이저
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # 검증
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Epoch {epoch+1}: val_acc={correct/total:.4f}")

# 모델 저장
torch.save(model.state_dict(), 'best_model.pth')