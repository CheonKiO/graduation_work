import json
from langchain_ollama import OllamaLLM
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from pydantic import BaseModel
from typing import Dict, List, Dict, Optional
from langchain_ollama import OllamaLLM
from deep_translator import GoogleTranslator

def translate_to_korean(text):
    translator = GoogleTranslator(source='auto', target='ko')
    translated_text = translator.translate(text)
    return translated_text

upper_category_mapping = {
    0: 'top', 
    1: 'bottom', 
    2: 'outer', 
    3: 'onepiece',
    4: 'unknown',
}

category_mapping = {
    0: 'top',
    1: 'blouse',
    2: 't-shirt',
    3: 'knitwear',
    4: 'shirt',
    5: 'bra-top',
    6: 'hoodie',
    7: 'vest',
    8: 'coat',
    9: 'jacket',
    10: 'jumper',
    11: 'padded-jacket',
    12: 'cardigan',
    13: 'zip-up',
    14: 'jeans',
    15: 'pants',
    16: 'skirt',
    17: 'leggings',
    18: 'jogger-pants',
    19: 'jumpsuit',
    20: 'dress',
    21: 'unknown',
}

color_mapping = {
    0: 'black',
    1: 'white',
    2: 'grey',
    3: 'red',
    4: 'pink',
    5: 'orange',
    6: 'beige',
    7: 'brown',
    8: 'yellow',
    9: 'green',
    10: 'khaki',
    11: 'mint',
    12: 'blue',
    13: 'navy',
    14: 'skyblue',
    15: 'purple',
    16: 'lavender',
    17: 'wine',
    18: 'neon',
    19: 'gold',
    20: 'silver',
    21: 'unknown',
}

material_mapping = {
    0: 'fur',
    1: 'mouton',
    2: 'suede',
    3: 'angora',
    4: 'corduroy',
    5: 'sequin/glitter',
    6: 'denim',
    7: 'jersey',
    8: 'tweed',
    9: 'velvet',
    10: 'vinyl/PVC',
    11: 'wool/cashmere',
    12: 'synthetic fiber',
    13: 'knit',
    14: 'lace',
    15: 'linen',
    16: 'mesh',
    17: 'fleece',
    18: 'neoprene',
    19: 'silk',
    20: 'spandex',
    21: 'jacquard',
    22: 'leather',
    23: 'cotton',
    24: 'chiffon',
    25: 'woven',
    26: 'padding',
    27: 'hair knit',
    28: 'unknown',
}

print_mapping = {
    0: 'check',
    1: 'stripe',
    2: 'zigzag',
    3: 'leopard',
    4: 'zebra',
    5: 'dot',
    6: 'camouflage',
    7: 'argyle',
    8: 'floral',
    9: 'skull',
    10: 'tie-dye',
    11: 'gradation',
    12: 'solid',
    13: 'graphic',
    14: 'houndstooth',
    15: 'gingham',
    16: 'lettering',
    17: 'paisley',
    18: 'mixed',
    19: 'heart',
    20: 'snake-skin',
    21: 'unknown',
}

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# Ollama LLM 설정 (Llama3.2 모델 사용 예시)
ollama_llm = OllamaLLM(model="llama3.2")

# 의류 아이템 모델 정의
class ClothesItem(BaseModel):
    upper_category: str
    category: str
    color: str
    print: str
    material: str

# 추천 결과 모델 정의
class RecommendationResult(BaseModel):
    recommendation: str  # 'yes' or 'no'
    score: int           # 0 to 10
    comment: str         # 피드백 코멘트

# 전체 응답 모델 정의
class ClosetResponse(BaseModel):
    recommendations: Dict[str, RecommendationResult]
    overall_review: str
    best_combination: List[str]


# 요청 모델 정의
class ClosetRequest(BaseModel):
    closet: Dict[str, ClothesItem]
    target: ClothesItem

# 역할과 응답 형식을 정의하는 프롬프트
def construct_prompt(closet: Dict[str, ClothesItem], target: ClothesItem) -> str:
    role_description = """
    You are a fashion assistant. Your role is to help the user decide whether the new item is a good purchase. 
    
    For each clothing item in the user's closet, provide a recommendation for the target item based on the following criteria:
    - You need to consider harmony with the "target" item.
    - please return the result like json{name:,recommendation:,score:,comment:}
    - Provide the recommendation using the following format:
        - Recommendation: [yes or no]
        - Score: [0 to 10]
        - Comment: [Explain why the clothing suits or does not suit the target.]
    -To convert to json, don't say anything other than the answer in json format.
    """

    closet_info = "\n".join([f"{key}: {item.model_dump()}" for key, item in closet.items()])
    target_info = f"Target item: {target.model_dump()}"

    full_prompt = f"{role_description}\n\nCloset:\n{closet_info}\n\n{target_info}\n\nAssistant:"
    return full_prompt


# Ollama에서 응답을 반환하는 엔드포인트 정의
@app.post("/ask", response_model=ClosetResponse)
async def ask_ollama(request: ClosetRequest):
    closet = request.closet
    target = request.target
    # 전처리: target에 따라 남길 항목 결정
    item_categories = {
        'top': ['bottom', 'outer'],         # 상의일 경우 남길 항목
        'bottom': ['top', 'outer'],         # 하의일 경우 남길 항목
        'outer': ['top', 'bottom', 'onepiece'],  # 아우터일 경우 남길 항목
        'onepiece': ['outer']                # 원피스일 경우 남길 항목
    }
    
    # 남길 아이템 목록 생성
    keep_categories = item_categories.get(target.upper_category, [])
    print(keep_categories)
    # 필터링된 closet 생성
    filtered_closet = {
        item: details for item, details in closet.items()
        if details.upper_category in keep_categories  # upper_category에 직접 접근
    }

    try:
        # 프롬프트 생성
        prompt = construct_prompt(filtered_closet, target)
        
        # LangChain Ollama 통합을 통해 프롬프트 전달 및 응답 받기
        raw_response = ollama_llm.invoke(prompt)
        print(raw_response)  # 디버깅을 위해 응답 출력
        
        # 응답 파싱
        overall_review = "No"
        best_combination = []
        recommendations = {}
        
        try:
            parsed_result = json.loads(raw_response)  # JSON 문자열을 파싱합니다.
            print("JSON parsing successful!")  # 파싱 성공 메시지 출력
        except json.JSONDecodeError as e:
            print("JSON parsing error:", e)  # 파싱 오류 메시지 출력
            return {
                "recommendations": {},
                "overall_review": "No",
                "best_combination": [],
                "error": "JSON parsing failed"
            }

        # 각 항목 정보 출력 및 recommendations 사전에 저장
        for item in parsed_result:
            print(item)
            item_name = item['name']
            recommendation = item['recommendation']  # recommendation 필드를 직접 사용
            score = item['score']  # Score에 직접 접근
            comment = item['comment']  # Comment에 직접 접근
            # 번역 실행
            print(comment)
            translated_text = translate_to_korean(comment)
            print(translated_text)
            # recommendations 사전에 저장
            recommendations[item_name] = {
                "Recommendation": recommendation,
                "Score": score,
                "Comment": comment
            }

        # Overall Recommendation 결정
        best_score = max(item['score'] for item in parsed_result)  # score에 직접 접근
        overall_combination_score = 0

        if target == 'top':
            overall_combination_score = sum(item['score'] for item in parsed_result if item['recommendation'] and item['name'] in ['item4', 'item6'])
        elif target == 'bottom':
            overall_combination_score = sum(item['score'] for item in parsed_result if item['recommendation'] and item['name'] in ['item2', 'item4'])
        elif target == 'outer':
            overall_combination_score = sum(item['score'] for item in parsed_result if item['recommendation'] and item['name'] in ['item6', 'item4'])
        elif target == 'onepiece':
            overall_combination_score = sum(item['score'] * 2 for item in parsed_result if item['recommendation'] and item['name'] in ['item8'])

        if best_score >= 14:
            overall_review = "Yes"

        # 결과 반환
        return ClosetResponse(
            recommendations=recommendations,
            overall_review=overall_review,
            best_combination=best_combination
        )
    
    except Exception as e:
        return {
            "recommendations": {},
            "overall_review": "No",
            "best_combination": [],
            "error": str(e)
        }




# 이미지 전처리 설정 (모델 학습 시 사용한 transforms와 일치시킴)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# UppermaterialModel 정의
class UppermaterialModel(nn.Module):
    def __init__(self, num_upper_categories, num_categories):
        super(UppermaterialModel, self).__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        self.fc_print = nn.Linear(512, num_upper_categories)
        self.fc_material = nn.Linear(512, num_categories)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        print_output = self.dropout(self.fc_print(x))
        material_output = self.dropout(self.fc_material(x))
        return {
            'print': print_output,
            'material': material_output
        }

# UpperCategoryCategoryModel 정의
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

# ColorModel 정의
class ColorModel(nn.Module):
    def __init__(self, num_colors):
        super(ColorModel, self).__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        self.fc_color = nn.Linear(512, num_colors)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        color_output = self.dropout(self.fc_color(x))
        return {
            'color': color_output,
        }

# 모델 로드 및 설정 (사전 학습된 가중치)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# UpperCategoryCategoryModel
num_upper_categories = 5  # 상위 카테고리 수
num_categories = 22  # 하위 카테고리 수
category_model = UpperCategoryCategoryModel(num_upper_categories, num_categories)
category_model.load_state_dict(torch.load('upper_category_category_model14.pth', map_location=device, weights_only=True))
category_model.to(device)
category_model.eval()

# UppermaterialModel
num_prints = 22  # 프린트 종류 수
num_materials = 29  # 소재 종류 수
uppermaterial_model = UppermaterialModel(num_prints, num_materials)
uppermaterial_model.load_state_dict(torch.load('print_material_model7.pth', map_location=device, weights_only=True))
uppermaterial_model.to(device)
uppermaterial_model.eval()

# ColorModel
num_colors = 22  # 색상 종류 수
color_model = ColorModel(num_colors)
color_model.load_state_dict(torch.load('color_model13.pth', map_location=device, weights_only=True))
color_model.to(device)
color_model.eval()

# 이미지 예측 함수
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    
    # 예측 실행
    with torch.no_grad():
        category_outputs = category_model(image)
        upper_category_pred = torch.argmax(category_outputs['upper_category'], dim=1).item()
        category_pred = torch.argmax(category_outputs['category'], dim=1).item()
        
        material_outputs = uppermaterial_model(image)
        print_pred = torch.argmax(material_outputs['print'], dim=1).item()
        material_pred = torch.argmax(material_outputs['material'], dim=1).item()
        
        color_outputs = color_model(image)
        color_pred = torch.argmax(color_outputs['color'], dim=1).item()
    
    return {
        "upper_category": upper_category_pred,
        "category": category_pred,
        "print": print_pred,
        "material": material_pred,
        "color": color_pred
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
