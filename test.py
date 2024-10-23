import requests
import json

# 요청을 보낼 서버의 URL
url = "http://127.0.0.1:8000/ask"

# 옷장 데이터 (closet)
closet_data = {
    "item1": {
        "upper_category": "top",
        "category": "t-shirt",
        "color": "navy",
        "print": "solid",
        "material": "cotton"
    },
    "item2": {
        "upper_category": "top",
        "category": "button-up shirt",
        "color": "white",
        "print": "striped",
        "material": "cotton"
    },
    "item3": {
        "upper_category": "top",
        "category": "sweater",
        "color": "grey",
        "print": "solid",
        "material": "wool"
    },
    "item4": {
        "upper_category": "bottom",
        "category": "jeans",
        "color": "blue",
        "print": "solid",
        "material": "denim"
    },
    "item5": {
        "upper_category": "bottom",
        "category": "shorts",
        "color": "beige",
        "print": "plaid",
        "material": "cotton"
    },
    "item6": {
        "upper_category": "outer",
        "category": "jacket",
        "color": "navy",
        "print": "solid",
        "material": "denim"
    },
    "item7": {
        "upper_category": "outer",
        "category": "coat",
        "color": "black",
        "print": "solid",
        "material": "wool"
    },
    "item8": {
        "upper_category": "onepiece",
        "category": "dress",
        "color": "red",
        "print": "floral",
        "material": "cotton"
    },
    "item9": {
        "upper_category": "bottom",
        "category": "skirt",
        "color": "black",
        "print": "solid",
        "material": "silk"
    },
    "item10": {
        "upper_category": "top",
        "category": "hoodie",
        "color": "green",
        "print": "graphic",
        "material": "cotton"
    }
}

# 구매할 옷 데이터 (target)
target_data = {
    "upper_category": "top",
    "category": "shirt",
    "color": "white",
    "print": "striped",
    "material": "cotton"
}

# POST 요청에 보낼 JSON 데이터
data = {
    "closet": closet_data,
    "target": target_data
}

# POST 요청 보내기
response = requests.post(url, json=data)

# 결과 출력
if response.status_code == 200:
    response_json = response.json()
    print(json.dumps(response_json, indent=4))  # JSON 데이터를 보기 좋게 출력
else:
    print(f"Error: {response.status_code}, {response.text}")

