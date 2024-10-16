
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn

# 다중 분류 모델 정의
class MultiOutputModel(nn.Module):
    def __init__(self, num_upper_categories, num_categories, num_colors, num_prints, num_materials):
        super(MultiOutputModel, self).__init__()
        # Pre-trained model 사용 (예: ResNet18)
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # ResNet의 마지막 FC 레이어 제거
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # 각 분류에 대해 별도의 FC 레이어 정의
        self.fc_upper_category = nn.Linear(512, num_upper_categories)  # 상위 카테고리 분류
        self.fc_category = nn.Linear(512, num_categories)  # 하위 카테고리 분류
        self.fc_color = nn.Linear(512, num_colors)  # 색상 분류
        self.fc_print = nn.Linear(512, num_prints)  # 프린트 분류
        self.fc_material = nn.Linear(512, num_materials)  # 소재 분류

    def forward(self, x):
        # 기본 모델을 통해 이미지 특성 추출
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # Flatten

        # 각 분류 작업에 대한 출력 생성
        upper_category_output = self.fc_upper_category(x)
        category_output = self.fc_category(x)
        color_output = self.fc_color(x)
        print_output = self.fc_print(x)
        material_output = self.fc_material(x)

        return {
            'upper_category': upper_category_output,
            'category': category_output,
            'color': color_output,
            'print': print_output,
            'material': material_output
        }