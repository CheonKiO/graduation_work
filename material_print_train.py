import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FashionDataset  # 사용자 정의 데이터셋
from tqdm import tqdm  # 모듈에서 함수만 임포트

import torchvision.models as models
from torchvision.models import ResNet18_Weights

print(torch.__version__)  # PyTorch 버전
print(torch.version.cuda)  # PyTorch에서 사용하는 CUDA 버전
print(f"cuda: {torch.cuda.is_available()}")
# 하이퍼파라미터
batch_size = 128
num_epochs = 5  # 원하는 추가 학습 epoch 수
learning_rate = 0.0003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 변환
transform = transforms.Compose([
    transforms.Resize((256, 256)),                      # 이미지 크기 조정
    transforms.RandomHorizontalFlip(p=0.3),             # 좌우 반전
    transforms.RandomVerticalFlip(p=0.3),               # 상하 반전 추가
    transforms.RandomRotation(degrees=(-45, 45)),              # 무작위 회전 (더 큰 범위)
    transforms.RandomApply([transforms.CenterCrop(200), transforms.Resize((256, 256))], p=0.3),  # 0.3 확률로 가운데 자르고 다시 256x256 리사이즈
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.2),  # 무작위 원근 변환
    transforms.ToTensor(),                              # 텐서 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 기준 정규화
])
transform2 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),                              # 텐서 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 기준 정규화
])

# 파일 경로 설정
image_dir = '원천데이터'
labels_json = 'labeling_data'

# 데이터셋 및 DataLoader
dataset = FashionDataset(image_dir=image_dir, label_dir=labels_json, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_dir = '../2.Validation/원천데이터'
valid_json = '../2.Validation/라벨링데이터'

dataset2 = FashionDataset(image_dir=valid_dir, label_dir=valid_json, transform=transform2)
valid_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=True, num_workers=4)

# 클래스 수 설정 (라벨 수에 따라 설정)
num_upper_categories = len(dataset.print_mapping)
num_categories = len(dataset.material_mapping)

# 모델 정의 (상위 카테고리와 하위 카테고리만)
class UppermaterialModel(nn.Module):
    def __init__(self, num_upper_categories, num_categories):
        super(UppermaterialModel, self).__init__()
        # Pre-trained model 사용 (예: ResNet18)
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # ResNet의 마지막 FC 레이어 제거
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # 각 분류에 대해 별도의 FC 레이어 정의
        self.fc_print = nn.Linear(512, num_upper_categories)
        self.fc_material = nn.Linear(512, num_categories)
        self.dropout = nn.Dropout(p=0.4)  # 드롭아웃을 개별 레이어로 정의

    def forward(self, x):
        # 기본 모델을 통해 이미지 특성 추출
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # Flatten

        # 각 분류 작업에 대한 출력 생성
        print_output = self.dropout(self.fc_print(x))
        material_output = self.dropout(self.fc_material(x))

        return {
            'print': print_output,
            'material': material_output
        }

# 모델 초기화 및 설정
model = UppermaterialModel(num_upper_categories, num_categories).to(device)
model.load_state_dict(torch.load('print_material_model6.pth', map_location=torch.device('cpu')))
model.to(device)

# 손실 함수 및 옵티마이저
criterion_print = nn.CrossEntropyLoss()
criterion_material = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 예시 검증 함수 수정
def validate_model(model, dataloader):
    model.eval()
    val_loss = 0.0
    correct_print = 0
    correct_material = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            
            # 라벨 처리
            print_labels = labels['print'].to(device)
            material_labels = labels['material'].to(device)
            
            # 모델 예측
            outputs = model(images)
            
            # 손실 계산
            loss_print = criterion_print(outputs['print'], print_labels)
            loss_material = criterion_material(outputs['material'], material_labels)
            
            # 총 손실
            total_loss = (loss_print + loss_material)
            val_loss += total_loss.item()
            
            # 정확도 계산
            _, pred_print = torch.max(outputs['print'], 1)
            _, pred_material = torch.max(outputs['material'], 1)
            
            correct_print += (pred_print == print_labels).sum().item()
            correct_material += (pred_material == material_labels).sum().item()
            total_samples += print_labels.size(0)
        
    val_loss /= len(dataloader)
    val_print_acc = correct_print / total_samples
    val_material_acc = correct_material / total_samples
    
    print(f"Validation Loss: {val_loss:.4f}, print Acc: {val_print_acc:.4f}, "
          f"material Acc: {val_material_acc:.4f}")
    
    model.train()

# 학습 함수 수정
def train_model(model, dataloader, criterion_print, criterion_material, optimizer, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_print = 0
        correct_material = 0
        total_samples = 0
        
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)
            
            # 라벨을 텐서로 변환
            print_labels = labels['print'].clone().detach().to(device) if isinstance(labels['print'], torch.Tensor) else torch.tensor(labels['print'], dtype=torch.long).to(device)
            material_labels = labels['material'].clone().detach().to(device) if isinstance(labels['material'], torch.Tensor) else torch.tensor(labels['material'], dtype=torch.long).to(device)

            # Forward pass
            outputs = model(images)
            
            # Loss computation for each output
            loss_print = criterion_print(outputs['print'], print_labels)
            loss_material = criterion_material(outputs['material'], material_labels)
            
            # Total loss
            total_loss = (loss_print + loss_material)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            # Predictions and accuracy calculation
            _, pred_print = torch.max(outputs['print'], 1)
            _, pred_material = torch.max(outputs['material'], 1)
            
            correct_print += (pred_print == print_labels).sum().item()
            correct_material += (pred_material == material_labels).sum().item()
            total_samples += print_labels.size(0)
        
        epoch_loss = running_loss / len(dataloader)
        print_acc = correct_print / total_samples
        material_acc = correct_material / total_samples
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
              f"print Acc: {print_acc:.4f}, material Acc: {material_acc:.4f}")
        validate_model(model, valid_loader)
        if (epoch+1)%5 == 0:
            # 학습이 완료된 모델 저장
            torch.save(model.state_dict(), f'print_material_model{7 + epoch//5}.pth')
            print(f'모델 저장 완료: print_material_model{7 + epoch//5}.pth')

# 학습 시작
train_model(model, train_loader, criterion_print, criterion_material, optimizer, num_epochs)


