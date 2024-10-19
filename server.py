from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# 모델 정의 (학습한 모델 클래스와 동일하게 유지)
class UpperCategoryCategoryModel(nn.Module):
    def __init__(self, num_upper_categories, num_categories):
        super(UpperCategoryCategoryModel, self).__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        self.fc_upper_category = nn.Linear(512, num_upper_categories)
        self.fc_category = nn.Linear(512, num_categories)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        upper_category_output = self.dropout(self.fc_upper_category(x))
        category_output = self.dropout(self.fc_category(x))
        return {
            'upper_category': upper_category_output,
            'category': category_output
        }

# 이미지 전처리 설정 (모델 학습 시 사용한 transforms와 일치시킴)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 모델 로드 (사전 학습된 가중치)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_upper_categories = 5  # 실제 모델의 상위 카테고리 수
num_categories = 22  # 실제 모델의 하위 카테고리 수
model = UpperCategoryCategoryModel(num_upper_categories, num_categories)
model.load_state_dict(torch.load('upper_category_category_model14.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()

# 이미지 예측 함수
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, upper_category_pred = torch.max(outputs['upper_category'], 1)
        _, category_pred = torch.max(outputs['category'], 1)

    return {
        "upper_category": upper_category_pred.item(),
        "category": category_pred.item()
    }

# 이미지 업로드 엔드포인트 (POST 요청)
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "Invalid image file"})
    
    predictions = predict_image(image)
    return {"predictions": predictions}

# 서버 시작 명령 (uvicorn을 통해 실행)
