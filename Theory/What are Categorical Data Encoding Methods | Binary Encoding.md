# What are Categorical Data Encoding Methods | Binary Encoding

딥러닝 모델은 숫자 입력만 처리할 수 있습니다. 따라서 범주형 데이터를 모델에 넣기 전에 **수치형**으로 변환해야 합니다. 이번 글에서는 다양한 인코딩 기법 중에서도 **이진 인코딩(Binary Encoding)** 에 집중해, 개념부터 코드 예제, 실제 딥러닝 모델 구성까지 차근차근 살펴보겠습니다.

***

## 1. 범주형 데이터와 인코딩의 필요성  
- **범주형 데이터(Categorical Data)**: 문자열이나 카테고리 형태로 제공되는 데이터입니다.  
- **명목형(Nominal)** vs. **서열형(Ordinal)**  
  - 명목형: 순서가 없고, ‘서울’, ‘부산’, ‘대구’처럼 각각 구분만 됩니다.  
  - 서열형: 순서가 의미를 갖습니다. 예컨대 학위(학사 < 석사 < 박사) 등이 있죠.  
- 인코딩 목적: 범주를 숫자 벡터로 바꿔 모델이 학습하도록 돕습니다.  
- 주요 기법: 라벨 인코딩, 원-핫 인코딩, 더미 인코딩, 해시 인코딩, 베이스N 인코딩, **이진 인코딩** 등  

***

## 2. 이진 인코딩(Binary Encoding)이란?  
1. 먼저 각 범주를 **정수**로 매핑(Label Encoding)합니다.  
2. 그 정수를 **2진수** 문자열로 변환합니다.  
3. 2진수의 각 자리(bit)를 **각 컬럼**으로 분리해 입력 피처로 사용합니다.  
4. 원-핫 인코딩보다 차원이 적고, 해시 인코딩보다 충돌 위험이 낮습니다.  

예를 들어, `{'A':0, 'B':1, 'C':2, 'D':3}`로 매핑 후 2진수로 바꾸면:  
- A → `0` → ``
- B → `1` → ``[1]
- C → `10` → `[1,0]`  
- D → `11` → `[1,1]`  

***

## 3. Python으로 이진 인코딩 적용하기  

```python
import pandas as pd
import category_encoders as ce

# 샘플 데이터
df = pd.DataFrame({
    'city': ['Seoul','Busan','Daegu','Seoul','Incheon','Busan','Gwangju']
})

# 이진 인코더 객체 생성
encoder = ce.BinaryEncoder(cols=['city'], return_df=True)

# 인코딩 수행
df_encoded = encoder.fit_transform(df)
print(df_encoded)
```

출력 예시:  
| city_0 | city_1 | city_2 |
|-------|--------|--------|
| 0     | 0      | 0      |
| 0     | 0      | 1      |
| 0     | 1      | 0      |
| 0     | 0      | 0      |
| 0     | 1      | 1      |
| 0     | 0      | 1      |
| 1     | 0      | 0      |

- `city_0`, `city_1`, `city_2`는 2진수 비트에 대응합니다.  
- 원본 `city` 컬럼은 제거되고, 대체되었습니다.  

***

## 4. 딥러닝 모델에 이진 인코딩 적용 예제  

TensorFlow + Keras를 사용해 간단한 분류 모델을 만들어 보겠습니다.  

```python
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 예시: 가상의 도시별 수요 예측 데이터
data = pd.DataFrame({
    'city': ['Seoul','Busan','Daegu','Incheon','Gwangju']*200,
    'demand': [100, 80, 60, 50, 30]*200
})

# 이진 인코딩 적용
encoder = ce.BinaryEncoder(cols=['city'], return_df=True)
data_enc = encoder.fit_transform(data[['city']])
X = data_enc.values
y = data['demand'].values

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # 회귀 문제
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
```

- **입력 레이어** 크기는 이진 인코딩으로 생성된 컬럼 수(`X_train.shape`)와 일치해야 합니다.[1]
- 회귀 문제로 설정했지만, 분류 문제라면 마지막 레이어와 손실 함수를 변경하면 됩니다.  

***

## 5. 이진 인코딩을 선택해야 할 때  
- **카디널리티(서로 다른 값의 개수)**가 높을 때  
- 원-핫 인코딩 시 차원이 너무 커질 때  
- 해시 인코딩의 충돌 위험을 줄이고 싶을 때  
- 수치 크기를 유지하며 순서를 고려하지 않을 때  

***

## 6. 주의 사항  
- 인코딩 후 각 비트 컬럼이 서로 상관관계를 갖습니다.  
- 스케일링이 필요할 수 있습니다(특히 신경망 모델에서).  
- 학습 데이터와 테스트 데이터에 **동일한 인코더**를 적용해야 합니다.  

***

## 마무리  
이진 인코딩은 대규모 범주형 변수를 처리할 때 유용합니다.  
딥러닝 모델에 바로 적용할 수 있고, 효율적인 차원 축소가 가능합니다.  
다양한 인코딩 기법 중 데이터와 모델 특성에 맞게 활용해 보세요.  
궁금한 점은 댓글로 남겨주세요. 감사합니다!

[1](https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/)
