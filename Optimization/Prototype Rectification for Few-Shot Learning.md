# Prototype Rectification for Few-Shot Learning | Image classification

## 1. 핵심 주장 및 주요 기여  
본 논문은 소수(1–5개) 샘플만으로 새로운 클래스를 분류하는 Few-Shot Learning 상황에서, 클래스 프로토타입이 **내부(intra-class) 및 클래스 간(cross-class) 바이어스**로 인해 실제 모든 샘플 분포를 대표하지 못한다는 문제를 규명한다. 제안된 **Bias Diminishing 모듈**은  
- **Intra-Class Bias Diminishing**: 라벨 예측 확신도가 높은 쿼리 샘플을 의사 라벨링(pseudo-labeling)으로 지원 집합에 추가하고, 기본 프로토타입과의 코사인 유사도를 기반으로 가중합하여 프로토타입을 보정  
- **Cross-Class Bias Diminishing**: 지원 집합과 쿼리 집합의 평균 특징 벡터 차이를 계산하여 쿼리 샘플에 이동(shift) 항 ξ를 더함  

을 통해 프로토타입이 실제 클래스 분포(mean)에 더 근접하도록 조정한다.  

주요 기여는 다음과 같다.  
1. Few-Shot 상황에서 프로토타입 편향을 유발하는 **Intra-Class / Cross-Class Bias** 개념 제시  
2. 단순·효율적인 **Bias Diminishing 모듈** 설계  
3. 이론적으로 **프로토타입 샘플 수 증가가 기대 성능 하한을 증가**시킨다는 증명  
4. miniImageNet·tieredImageNet·Meta-Dataset 에서 SOTA 달성  

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계  

### 문제 정의  
Few-Shot 상황에서 클래스별 K개의 지원 샘플만으로 평균 벡터(프로토타입)를 계산하면,  
- intra-class bias: 실제 클래스 전체 분포의 평균과 지원 샘플 평균 차이  
- cross-class bias: 여러 클래스의 지원·쿼리 분포 간 평균 차이  
로 인해 대표성이 떨어져 분류 성능이 저하됨.  

### 제안 방법  
1. **CSPN (Cosine Similarity Based Prototypical Network)**  
   -  베이스 클래스 다중 분류 학습 시 **Cosine Classifier**(식 (1))로 특징 학습  
   -  Few-Shot 단계에서 지원 샘플 K개 평균으로 기본 프로토타입 $$P_n = \frac1K\sum_i X_{i,n}$$ 산출(식 (3))  
2. **Intra-Class Bias Diminishing**  
   -  쿼리 집합에서 예측 확신도 상위 Z개 샘플 의사 라벨링 후 지원 집합에 추가  
   -  코사인 유사도 기반 가중합(proto rectification):  

```math
       w_{i,n}=\frac{\exp(\varepsilon\cdot\cos(X'_{i,n},P_n))}{\sum_j\exp(\varepsilon\cdot\cos(X'_{j,n},P_n))},\quad
       P'_n=\sum_i w_{i,n}X'_{i,n}
```  

3. **Cross-Class Bias Diminishing**  
   -  지원·쿼리 집합 평균 차이를 이동 항 $$\displaystyle \xi = \frac1{|S|}\sum_{x\in S}x - \frac1{|Q|}\sum_{x\in Q}x$$로 정의(식 (8))  
   -  모든 쿼리 특징에 ξ 더하여 지원 분포로 정렬  
4. **이론적 분석**  
   -  프로토타입 샘플 수 $$T=K+Z$$ 증가 시 코사인 유사도 기대값 하한 상승(식 (16))  
   -  Cross-Class 이동 항 유도(식 (24))  

### 모델 구조  
- **Backbone**: WRN-28-10, ResNet-12, ConvNet 계열  
- **Feature Extractor** → **Cosine Classifier** 학습 → Few-Shot 단계 CSPN + BD 모듈  

### 성능 향상  
- miniImageNet 1-shot: **61.84%→70.31%**, 5-shot: **78.64%→81.89%**  
- tieredImageNet 1-shot: **69.20%→78.74%**, 5-shot: **84.31%→86.92%**  
- Meta-Dataset 5-shot 평균 랭크 1.9위  

### 한계  
- **Transductive setting**(모든 쿼리를 한꺼번에 사용) 가정하므로, 온라인·실시간 예측 시 제약  
- 의사 라벨링된 샘플이 많아질수록 잘못된 추가가 성능 저하 유발 가능  
- 쿼리 분포 이동이 클래스 간 경계 왜곡 초래 여부 추가 연구 필요  

## 3. 일반화 성능 향상 관점  
의사 라벨링으로 **유효 샘플 수**를 늘려 프로토타입 추정 편향을 낮추는 전략은, 클래스·도메인 불균형이 심한 환경에서도  
- 특징 공간에서 클러스터 응집도 강화  
- 지원-쿼리 도메인 정렬 효과  
를 기대할 수 있어, **도메인 적응(domain adaptation)** 및 **장교차(over-cross) 클래스 일반화** 과제에도 확장 적용 가능하다.

## 4. 향후 연구에 미치는 영향 및 고려 사항  
- **의사 라벨링 신뢰도 제고**: 적응형 임계값·불확실성 기반 샘플 선택 연구  
- **온라인·인크리멘털 확대**: 트랜스덕티브 전제 완화, 스트리밍 데이터 적용 방안  
- **다중 도메인 Few-Shot**: 여러 지원 도메인 간 크로스-바이어스 정량화 및 보정  
- **안전성·강인성**: 잘못된 의사 라벨이 모델에 미치는 영향 제어, 불확실성 추정 강화  

이러한 고려를 바탕으로, 본 논문의 **Bias Diminishing** 개념은 Few-Shot 일반화 연구 전반에 새로운 방향성을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e88b6872-3ff1-4af2-a15a-69271898447b/1911.10713v4.pdf
