import os
import json
from torch.utils.data import Dataset
import pickle
from PIL import Image
from tqdm import tqdm

# 데이터셋 클래스 정의
class FashionDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_labels = []
        self.upper_category_mapping = {
            '상의':0, 
            '하의':1, 
            '아우터':2, 
            '원피스':3,
            '미상': 4,
        }
        self.category_mapping = {
            '탑': 0,
            '블라우스': 1,
            '티셔츠': 2,
            '니트웨어': 3,
            '셔츠': 4,
            '브라탑': 5,
            '후드티': 6,
            '베스트': 7,
            '코트': 8,
            '재킷': 9,
            '점퍼': 10,
            '패딩': 11,
            '가디건': 12,
            '짚업': 13,
            '청바지': 14,
            '팬츠': 15,
            '스커트': 16,
            '래깅스': 17,
            '조거팬츠': 18,
            '점프수트': 19,
            '드레스': 20,
            '미상': 21,
        }
        self.color_mapping = {
            '블랙': 0,
            '화이트': 1,
            '그레이': 2,
            '레드': 3,
            '핑크': 4,
            '오렌지': 5,
            '베이지': 6,
            '브라운': 7,
            '옐로우': 8,
            '그린': 9,
            '카키': 10,
            '민트': 11,
            '블루': 12,
            '네이비': 13,
            '스카이블루': 14,
            '퍼플': 15,
            '라벤더': 16,
            '와인': 17,
            '네온': 18,
            '골드': 19,
            '실버': 20,
            '미상': 21,
        }
        self.material_mapping = {
            '퍼': 0,
            '무스탕': 1,
            '스웨이드': 2,
            '앙고라': 3,
            '코듀로이': 4,
            '시퀸/글리터': 5,
            '데님': 6,
            '저지': 7,
            '트위드': 8,
            '벨벳': 9,
            '비닐/PVC': 10,
            '울/캐시미어': 11,
            '합성섬유': 12,
            '니트': 13,
            '레이스': 14,
            '린넨': 15,
            '메시': 16,
            '플리스': 17,
            '네오프렌': 18,
            '실크': 19,
            '스판덱스': 20,
            '자카드': 21,
            '가죽': 22,
            '면': 23,
            '시폰': 24,
            '우븐': 25,
            '패딩': 26,
            '헤어 니트': 27,
            '미상': 28,
        }
        self.print_mapping = {
            '체크': 0,
            '스트라이프': 1,
            '지그재그': 2,
            '호피': 3,
            '지브라': 4,
            '도트': 5,
            '카무플라쥬': 6,
            '아가일': 7,
            '플로럴': 8,
            '해골': 9,
            '타이다이': 10,
            '그라데이션': 11,
            '무지': 12,
            '그래픽': 13,
            '하운즈\xa0투스': 14,
            '깅엄': 15,
            '레터링': 16,
            '페이즐리': 17,
            '믹스': 18,
            '하트': 19,
            '뱀피': 20,
            '미상': 21,
        }
        if label_dir:
            self.load_or_create_dataset(label_dir)
        else:
            pass

    def load_or_create_dataset(self, label_dir):
        for item in os.listdir(label_dir):
            item_path = os.path.join(label_dir, item)
            if os.path.isdir(item_path):
                pickle_path = os.path.join(label_dir, f'{item}.pkl')

                if os.path.exists(pickle_path):
                    print(f"Loading dataset from {pickle_path}")
                    with open(pickle_path, 'rb') as f:
                        image_labels = pickle.load(f)
                    self.image_labels.extend(image_labels)
                else:
                    print(f"Processing {item}...")
                    for file_name in tqdm(os.listdir(item_path), desc=f"Loading files from {item}"):
                        if file_name.endswith('.json'):
                            json_path = os.path.join(item_path, file_name)
                            image_id = json.load(open(json_path))['이미지 정보']['이미지 식별자']
                            image_file = f"{image_id}.jpg"
                            image_path = os.path.join(self.image_dir, item, image_file)

                            if os.path.exists(image_path):
                                with open(json_path, 'r') as f:
                                    data = json.load(f)
                                categories = ['아우터', '상의', '하의', '원피스']
                                label_info = {}

                                for category in categories:
                                    if data['데이터셋 정보']['데이터셋 상세설명']['라벨링'].get(category):
                                        category_data = data['데이터셋 정보']['데이터셋 상세설명']['라벨링'][category]
                                        if bool(category_data[0]):
                                            label_info = {
                                                '상위 카테고리': category,
                                                '카테고리': category_data[0].get('카테고리', '미상'),
                                                '색상': category_data[0].get('색상', '미상'),
                                                '프린트': category_data[0].get('프린트', []),
                                                '소재': category_data[0].get('소재', [])
                                            }
                                            self.image_labels.append((image_path, label_info))
                            else:
                                print(f"Image not found: {image_path}")
        # self.image_labels를 한 번에 저장하지 않고 필요한 데이터만 로딩

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_path, label_data = self.image_labels[idx]

        # 이미지 로딩 및 전처리
        image = self.load_image(image_path)
        if self.transform:
            image = self.transform(image)
        # 라벨 처리
        upper_category_label = self.get_upper_category_tensor(label_data)
        category_label = self.get_category_tensor(label_data)
        color_label = self.get_color_tensor(label_data)
        print_label = self.get_print_tensor(label_data)
        material_label = self.get_material_tensor(label_data)
        labels = {
            'upper_category': upper_category_label,
            'category': category_label,
            'color': color_label,
            'print': print_label,
            'material': material_label
        }
        return image, labels


    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image
    
    def get_upper_category_tensor(self, label_data):
        upper_category_name = label_data.get('상위 카테고리', '미상')
        return self.upper_category_mapping.get(upper_category_name,-1)

    def get_category_tensor(self, label_data):
        category_name = label_data.get('카테고리', '미상')
        return self.category_mapping.get(category_name, -1)

    def get_color_tensor(self, label_data):
        color_name = label_data.get('색상', '미상')
        return self.color_mapping.get(color_name, -1)
        

    def get_print_tensor(self, label_data):
        print_name = label_data.get('프린트','미상')
        if len(print_name) == 0:
            return self.print_mapping['미상']
        if isinstance(print_name, list) and len(print_name) > 0:
            print_name = print_name[0]
        return self.print_mapping.get(print_name, -1)

    def get_material_tensor(self, label_data):
        material_name = label_data.get('소재', '미상')
        if len(material_name) == 0:
            return self.print_mapping['미상']
        if isinstance(material_name, list) and len(material_name) > 0:
            material_name = material_name[0]
        return self.material_mapping.get(material_name, -1)