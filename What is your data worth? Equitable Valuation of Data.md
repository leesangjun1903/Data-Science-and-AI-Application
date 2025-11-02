# What is your data worth? Equitable Valuation of Data

## 1. 핵심 주장과 주요 기여

이 논문은 머신러닝 시스템에서 **개별 데이터의 가치를 공정하게 측정하는 문제**를 다룹니다. 데이터가 경제적 성장의 연료가 되면서, 개인이 생성한 데이터에 대한 보상이 필요하지만 공정한 평가 방법이 없다는 문제에서 출발합니다.[1]

**핵심 기여**는 다음과 같습니다:[1]

1. **Data Shapley 프레임워크 제안**: 게임 이론의 Shapley value를 데이터 평가에 적용하여, 각 훈련 데이터가 예측기 성능에 기여하는 가치를 정량화하는 원칙적인 방법을 제시
2. **세 가지 공정성 원칙 만족**: Data Shapley는 공정한 데이터 평가를 위한 세 가지 자연스러운 속성을 **유일하게** 만족
3. **효율적인 근사 알고리즘 개발**: 복잡한 학습 알고리즘과 대규모 데이터셋에서 사용 가능한 TMC-Shapley(Truncated Monte Carlo)와 G-Shapley(Gradient-based) 근사 방법 제시
4. **실용적 응용 시연**: 저품질 데이터 탐지, 노이즈/오염 데이터 식별, 새로운 데이터 수집 가이드, 도메인 적응 등 다양한 실제 응용 사례 제시

## 2. 해결하고자 하는 문제와 제안 방법

### 문제 정의

지도 학습 환경에서, $$n$$개의 훈련 데이터 $$D = \{(x_i, y_i)\}_{i=1}^n$$가 학습 알고리즘 $$\mathcal{A}$$로 모델을 학습하고, 성능 지표 $$V$$로 평가될 때, 각 데이터 소스의 가치 $$\phi_i(D, \mathcal{A}, V)$$를 **공정하게** 결정하는 것이 목표입니다.[1]

기존의 **Leave-One-Out(LOO)** 방법은 $$\phi_i = V(D) - V(D - \{i\})$$로 계산하지만, 공정성 조건을 만족하지 못합니다.[1]

### 공정성 조건

공정한 데이터 평가는 다음 세 가지 속성을 만족해야 합니다:[1]

1. **Zero Property**: 데이터 $$i$$가 어떤 부분집합에 추가되어도 성능을 변화시키지 않으면 가치는 0
2. **Symmetry**: 데이터 $$i$$와 $$j$$가 모든 부분집합에서 동일한 기여를 하면 동일한 가치 할당
3. **Linearity**: 성능 지표가 합으로 표현될 때, 가치도 각 지표에 대한 가치의 합: $$\phi_i(V + W) = \phi_i(V) + \phi_i(W)$$

### Data Shapley 정의

위 세 가지 조건을 만족하는 유일한 형태는 다음과 같습니다:[1]

$$
\phi_i = C \sum_{S \subseteq D - \{i\}} \frac{V(S \cup \{i\}) - V(S)}{\binom{n-1}{|S|}}
$$

여기서 합은 데이터 $$i$$를 포함하지 않는 모든 부분집합 $$S$$에 대한 것이며, $$C$$는 임의의 상수입니다. 이를 **Data Shapley value**라고 명명합니다.[1]

이는 협력 게임 이론의 Shapley value와 동일한 형태로, 각 데이터를 플레이어로, 학습 알고리즘을 통한 협력으로 얻는 보상을 성능으로 보는 관점입니다.[1]

### 근사 알고리즘

**Equation 1**의 직접 계산은 $$2^n$$개의 부분집합을 고려해야 하므로 계산적으로 불가능합니다. 이를 해결하기 위해 두 가지 근사 방법을 제안합니다:[1]

#### 1) TMC-Shapley (Truncated Monte Carlo Shapley)

Monte Carlo 방법과 truncation을 결합한 알고리즘입니다:[1]

- 데이터의 무작위 순열(permutation) 샘플링
- 순열을 따라 데이터를 순차적으로 추가하며 marginal contribution $$V(S \cup \{i\}) - V(S)$$ 계산
- 성능이 수렴하면 계산 중단(truncation)하여 계산량 절감
- 여러 순열에 대한 평균으로 Shapley value 추정

알고리즘 1에서 "Performance Tolerance" 기준으로 $$|V(D) - v_j^t| <$$ 허용치일 때 truncation하여 나머지 marginal contribution을 0으로 근사합니다[1]. Appendix E에서 25% truncation이 truncation 없는 경우와 0.8의 rank correlation을 보여 효율성과 정확성의 균형을 입증했습니다[1].

#### 2) G-Shapley (Gradient Shapley)

확률적 경사 하강법(SGD) 기반 학습 알고리즘에 특화된 근사 방법입니다:[1]

- 한 epoch만 학습하는 것으로 완전히 학습된 모델을 근사
- 무작위 순열에 따라 한 번에 하나의 데이터로 gradient descent 수행: $$\theta_j^t \leftarrow \theta_{j-1}^t - \alpha \nabla_\theta L(\pi^t[j]; \theta_{j-1})$$
- 각 단계에서 성능 변화가 marginal contribution

합성 데이터에서 TMC-Shapley와 true Shapley value 간 Pearson correlation이 98.4-99.5%로 높은 정확도를 보였습니다.[1]

## 3. 모델 구조와 성능 향상

### 모델 구조

Data Shapley는 **모델에 구애받지 않는(model-agnostic)** 프레임워크입니다. 학습 알고리즘 $$\mathcal{A}$$를 블랙박스로 취급하며, 다음 실험에서 다양한 모델에 적용되었습니다:[1]

- Logistic Regression
- Naive Bayes
- Random Forest
- Convolutional Neural Networks
- Inception-V3 (fine-tuning)
- LSTM (DeepTag)

### 성능 향상 실증

#### 1) 저품질 데이터 탐지

**Mislabeled Data 탐지**: Spam classification(20% mislabeled), Flower classification(10% mislabeled), Fashion MNIST(10% mislabeled) 실험에서, Data Shapley는 LOO보다 적은 검사로 잘못된 라벨을 발견했습니다. 특히 neural network 모델에서 LOO는 random inspection과 유사한 성능을 보인 반면, Data Shapley는 효과적으로 mislabeled data를 낮은 가치로 할당했습니다.[1]

**Noisy Data 탐지**: Dog vs Fish 데이터셋에서 10% 이미지에 Gaussian noise 추가 시, noise level이 증가할수록 noisy 이미지의 평균 Shapley value가 감소했습니다(noise level 0.5에서 음수값).[1]

#### 2) 모델 성능 개선

**고가치 데이터 제거**: UK Biobank의 유방암 및 피부암 예측 태스크에서, Shapley value가 높은 데이터를 순서대로 제거하면 성능이 급격히 저하되어 고가치 데이터의 중요성을 입증했습니다.[1]

**저가치 데이터 제거**: 역순으로 낮은 Shapley value 데이터를 제거하면 오히려 성능이 향상되어, 일부 데이터가 모델에 해롭다는 것을 보여줍니다.[1]

**새로운 데이터 획득**: 2000명의 후보 환자 pool에서, 고가치 훈련 데이터와 유사한 환자를 Random Forest로 추정하여 순차적으로 추가하면, 무작위 추가나 LOO보다 효과적으로 성능이 증가했습니다.[1]

#### 3) 도메인 적응(Domain Adaptation)

훈련 데이터와 테스트 데이터의 분포가 다를 때, target data에서 평가한 Data Shapley로 음수 가치 데이터를 제거하고 양수 가치로 가중치를 조정하여 재학습한 결과:[1]

| Source → Target | Task | Original | Adapted |
|---|---|---|---|
| Google → HAM10000 | Skin Lesion Classification | 29.6% | 37.8% |
| CSU → PP | Disease Coding | 87.5% | 90.1% |
| LFW+ → PPB | Gender Detection | 84.1% | 91.5% |
| MNIST → USPS | Digit Recognition | 30.8% | 39.1% |
| Email → SMS | Spam Detection | 68.4% | 86.4% |

특히 LFW+에서 PPB로의 성별 탐지 적응에서, 음수 가치를 받은 이미지는 모두 남성(과대표집단)이었고, 상위 20% 고가치 이미지는 여성이었습니다.[1]

## 4. 일반화 성능 향상 가능성

### 직접적 일반화 효과

**분포 불일치 해결**: 도메인 적응 실험에서 입증되었듯이, target 분포에서 평가한 Data Shapley는 source 데이터 중 일반화에 해로운 데이터를 식별합니다. 예를 들어, UK Biobank의 Nottingham 센터 데이터가 colon cancer 예측에서 음수 가치를 받은 이유는 해당 지역 환자의 나이 분포가 일반 인구와 달라, 나이라는 핵심 예측 feature와의 관계가 일반적이지 않았기 때문입니다.[1]

**Outlier 및 Corruption 제거**: Data Shapley는 낮은 가치를 통해 outlier와 데이터 오염을 효과적으로 포착합니다. 이러한 데이터를 제거하면 모델이 진정한 패턴에 집중하여 일반화 성능이 향상됩니다. 실제로 저가치 데이터 제거 시 held-out set 성능이 향상되었습니다.[1]

### 간접적 일반화 효과

**데이터 수집 전략**: 고가치 데이터의 특성을 학습하여 유사한 새 데이터를 수집하면, 모델의 일반화에 도움이 되는 데이터를 선별적으로 확보할 수 있습니다. 실험에서 Random Forest로 가치를 예측하여 고가치 유사 데이터를 추가한 결과, 성능이 체계적으로 증가했습니다.[1]

**Marginal Contribution의 학습 곡선**: TMC-Shapley의 truncation 기법은 학습 이론의 원리를 반영합니다. 훈련 데이터 크기가 증가하면 추가 데이터의 marginal contribution이 감소하는데, 이는 학습 곡선의 수렴을 나타냅니다. 이러한 통찰은 데이터 효율적인 학습 전략 수립에 활용될 수 있습니다.[1]

### 일반화의 한계

**Context 의존성**: Data Shapley value는 학습 알고리즘, 성능 지표, 그리고 다른 훈련 데이터에 의존합니다. 따라서 한 모델에서 높은 가치를 가진 데이터가 다른 모델에서는 낮을 수 있습니다. 실제로 서로 다른 모델 간 Shapley value의 rank correlation은 평균 0.32-0.52로 중간 수준이었습니다.[1]

**비선형 관계에서의 모델 미스매치**: 합성 데이터 실험에서, 비선형 feature-label 관계에서 비선형 모델의 성능을 향상시키는 데이터가 선형 모델에는 해로울 수 있어, Shapley value가 모델에 의존적임을 보여줍니다.[1]

## 5. 한계

### 계산 복잡도

**여전히 높은 계산 비용**: TMC-Shapley와 G-Shapley로 근사하더라도, 많은 순열을 샘플링하고 각 순열마다 모델을 재학습해야 하므로 대규모 데이터셋에서는 계산 비용이 여전히 높습니다. Truncation으로 일부 완화되지만 근본적인 해결책은 아닙니다.[1]

**근사 오차**: G-Shapley와 TMC-Shapley의 correlation은 모델과 데이터에 따라 0.57-0.97로 변동이 큽니다. 복잡한 neural network에서는 correlation이 감소하는 경향이 있습니다(0.7-0.8 range).[1]

### 개념적 한계

**공정성 조건의 타당성**: 논문은 세 가지 공정성 조건이 모든 ML 환경에 적절하지 않을 수 있으며, 다른 속성이 필요한 시나리오가 있을 수 있음을 인정합니다. 어떤 환경에 어떤 공정성 개념이 적절한지 명확히 이해하는 것이 향후 연구 과제입니다.[1]

**내재적 가치 미반영**: Data Shapley는 지도 학습의 훈련 성능에만 초점을 맞추며, 개인 데이터의 내재적 가치(프라이버시, 개인적 연관성 등)는 포착하지 못합니다. 논문은 사람들이 정확히 Shapley value만큼 보상받아야 한다고 제안하는 것이 아니라, 정량적 통찰을 제공하는 도구로 봐야 한다고 명시합니다.[1]

**테스트 세트 의존성**: 성능 지표 $$V$$는 유한한 테스트 세트에서 측정되므로, 진정한 일반화 성능의 근사치일 뿐입니다. 테스트 세트 선택이 Shapley value에 영향을 미칠 수 있습니다.[1]

### 실용적 한계

**소규모 데이터에만 정확한 검증**: 진정한 Shapley value와의 비교는 4-14개 데이터 포인트에서만 수행되었습니다. 실제 대규모 데이터셋에서의 근사 품질은 간접적으로만 평가됩니다.[1]

**Hyperparameter 민감성**: G-Shapley는 one-epoch 학습에 최적화된 learning rate를 찾기 위한 hyperparameter search가 필요하며, 이는 추가 계산 비용을 발생시킵니다.[1]

## 6. 향후 연구에 미치는 영향

### 이론적 영향

**데이터 중심 AI의 기초**: Data Shapley는 "데이터의 가치"라는 추상적 개념에 수학적으로 엄밀한 정의를 제공하여, 데이터 중심 AI(Data-Centric AI) 연구의 이론적 토대를 마련했습니다. 이는 데이터 품질, 데이터 선택, 데이터 획득 전략 연구에 원칙적 프레임워크를 제공합니다.[1]

**게임 이론과 ML의 융합**: Shapley value의 ML 적용은 게임 이론과 머신러닝의 새로운 연결고리를 제시하며, 협력적 학습(Federated Learning, Collaborative Learning) 환경에서의 기여도 평가 연구로 확장될 수 있습니다.[1]

### 실용적 영향

**데이터 시장(Data Marketplace)**: Data Shapley는 데이터 거래 플랫폼에서 공정한 가격 책정 메커니즘의 기반이 될 수 있습니다. 병렬 연구들이 Shapley value 기반 데이터 시장을 탐구하고 있습니다.[1]

**AutoML 및 데이터 큐레이션**: 자동화된 머신러닝 파이프라인에서 Data Shapley를 사용하여 데이터 정제, 이상치 탐지, 능동 학습(Active Learning) 전략을 자동화할 수 있습니다.[1]

**연합 학습(Federated Learning)**: 분산 환경에서 각 참여자의 데이터 기여도를 평가하여 인센티브 메커니즘을 설계하는 데 활용될 수 있습니다.

## 7. 향후 연구 시 고려사항

### 알고리즘 개선

**효율성 향상**: 더 효율적인 Shapley value 근사 방법 개발이 필요합니다. 특히 deep learning 모델과 대규모 데이터셋에서 실용적으로 사용 가능한 수준으로 계산 비용을 줄여야 합니다.[1]

**적응적 샘플링**: 모든 순열을 동등하게 샘플링하는 대신, 중요한 순열에 더 많은 샘플을 할당하는 적응적 Monte Carlo 방법을 고려해야 합니다.

### 이론적 확장

**공정성 공리의 확장**: 다양한 ML 시나리오(강화 학습, 비지도 학습, 온라인 학습 등)에 적합한 데이터 가치 평가 공리를 연구해야 합니다.[1]

**모델 불변성 연구**: 데이터 가치가 학습 알고리즘과 성능 지표에 어떻게 의존하는지 체계적으로 이해하고, 가능하면 모델에 덜 의존적인 가치 측정 방법을 탐구해야 합니다.[1]

### 응용 연구

**프라이버시 고려**: Data Shapley를 차등 프라이버시(Differential Privacy)와 통합하여, 개인 정보 보호를 유지하면서 데이터 가치를 평가하는 방법을 연구해야 합니다.

**동적 환경**: 데이터와 모델이 시간에 따라 변화하는 환경(continual learning, streaming data)에서 Data Shapley를 어떻게 적용할지 연구가 필요합니다.

**다중 이해관계자**: 여러 성능 지표와 이해관계자가 있는 환경에서 공정한 데이터 가치 평가 방법을 개발해야 합니다.

### 실증적 검증

**더 다양한 도메인**: 자연어 처리, 추천 시스템, 시계열 예측 등 다양한 도메인에서 Data Shapley의 효과를 검증해야 합니다.

**실제 데이터 시장 테스트**: 실제 데이터 거래 환경에서 Data Shapley 기반 가격 책정 메커니즘을 시범 운영하고 경제적 타당성을 검증해야 합니다.

이 논문은 데이터의 가치를 정량화하는 원칙적이고 실용적인 프레임워크를 제시함으로써, 데이터 중심 AI 시대의 중요한 이정표가 되었습니다. 특히 일반화 성능 향상에 기여하는 데이터를 식별하고, 도메인 적응 문제를 해결하는 데 효과적임을 입증했습니다. 향후 연구는 계산 효율성 개선, 이론적 확장, 그리고 다양한 실세계 응용으로 나아가야 할 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/00375612-21c8-41fc-a6df-6c14fa149c76/1904.02868v2.pdf)
