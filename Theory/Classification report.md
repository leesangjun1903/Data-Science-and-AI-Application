# 분류 모델 평가 지표 가이드: Classification Report부터 ROC Curve까지

**분류 모델 평가 지표**를 정리했습니다.  
각 지표의 원리와 해석 방법, 사이킷런 예시 코드를 함께 알아봅니다.

***

## 1. 혼동 행렬(Confusion Matrix)

혼동 행렬은 모델이 예측한 클래스와 실제 클래스 간의 관계를 행렬 형태로 보여줍니다.  
이진 분류 기준으로 다음 네 가지로 구분합니다.

- **TN (True Negative)**: 음성을 음성으로 맞춤  
- **TP (True Positive)**: 양성을 양성으로 맞춤  
- **FN (False Negative)**: 양성을 음성으로 잘못 분류  
- **FP (False Positive)**: 음성을 양성으로 잘못 분류  

```python
from sklearn.metrics import confusion_matrix

# y_test: 실제 레이블, y_pred: 예측 레이블
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

예시 출력:
```
[[TN, FP],
 [FN, TP]]
```

***

## 2. 정확도(Accuracy)

전체 샘플 중 올바르게 예측한 비율입니다.  

$$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} $$

정리: 모든 예측에서 맞춘 비율을 직관적으로 보여줍니다.

***

## 3. 정밀도(Precision)와 재현율(Recall)

양성(POSITIVE) 예측 성능을 더 세밀히 평가합니다.

- **Precision (양성 예측도)**  
  예측을 양성으로 한 샘플 중 실제 양성 비율  

$$ Precision = \frac{TP}{TP + FP} $$

- **Recall (재현율, 민감도)**  
  실제 양성 샘플 중 모델이 양성으로 맞춘 비율  

$$ Recall = \frac{TP}{TP + FN} $$

> **언제 중요한가?**  
> - 재현율이 중요할 때: 암 진단, 금융 사기 탐지  
> - 정밀도가 중요할 때: 스팸 필터링, 불필요 알림 방지  

```python
from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_test, y_pred))
print("Recall:   ", recall_score(y_test, y_pred))
```

***

## 4. F1 Score

Precision과 Recall의 조화 평균입니다.  
두 지표 간 불균형이 클수록 낮은 값에 더 가깝습니다.  
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} $$

```python
from sklearn.metrics import f1_score

print("F1 Score:", f1_score(y_test, y_pred))
```

***

## 5. Classification Report

사이킷런의 `classification_report`는 주요 지표를 한 번에 보여줍니다.  

```python
from sklearn.metrics import classification_report

target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))
```

출력 예시:
```
              precision  recall  f1-score  support

     class 0       0.50    1.00      0.67        1
     class 1       0.00    0.00      0.00        1
     class 2       1.00    0.67      0.80        3

    accuracy                           0.60        5
   macro avg       0.50    0.56      0.49        5
weighted avg       0.70    0.60      0.61        5
```

- **support**: 실제 샘플 개수  
- **macro avg**: 클래스별 지표 단순 평균  
- **weighted avg**: support 기준 가중 평균

***

## 6. ROC Curve와 AUC

ROC(Receiver Operating Characteristic) 곡선은 **False Positive Rate(FPR)** 대비 **True Positive Rate(TPR)** 변화를 시각화합니다.  
- TPR = Recall  
- FPR = $$\frac{FP}{FP + TN}$$

AUC(Area Under Curve)는 ROC 곡선 아래 면적을 의미하며, 0.5에서 1.0 사이입니다.  

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_score = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

- **임계값(threshold)** 조정으로 Precision/Recall trade-off를 관리할 수 있습니다.

***

## 7. 실전 예시: 타이타닉 생존 예측

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 데이터 로드 및 전처리
df = pd.read_csv('titanic_train.csv')
X = df.drop('Survived', axis=1)
y = df['Survived']

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# 모델 학습
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 평가
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Dead', 'Survived']))
```

### 마무리

이 가이드에서는 **Confusion Matrix**, **Accuracy**, **Precision**, **Recall**, **F1 Score**, **Classification Report**, **ROC Curve**를 학습했습니다.  
각 지표를 이해하고 활용하면 분류 모델의 성능을 다각도로 평가할 수 있습니다.  
여러 지표를 함께 살펴보며 모델 특성에 맞게 **임계값 조정**까지 시도해보세요!

[1](https://songseungwon.tistory.com/95)
[2](https://pycode.tistory.com/40)

- https://songseungwon.tistory.com/95
- https://pycode.tistory.com/40
- https://m.blog.naver.com/hannaurora/222498671200
