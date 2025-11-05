# Learning representations by back-propagating errors

### 1. 핵심 주장과 주요 기여 (간결 요약)

이 논문은 **역전파(backpropagation) 알고리즘**을 통해 다층 신경망에서 **숨겨진 층(hidden layer)의 내부 표현을 자동으로 학습**할 수 있음을 보여주었습니다. 주요 기여는 세 가지입니다:[1]

1. **신용할당 문제(Credit Assignment Problem) 해결**: 네트워크의 출력 오류를 역방향으로 전파하여 각 가중치가 전체 오류에 미친 영향을 정량화했습니다.[1]

2. **다층 네트워크의 학습 가능성 입증**: 종전에는 불가능하다고 여겨진 다층 신경망을 효과적으로 학습할 수 있음을 실험적으로 증명했습니다.[2]

3. **특성 표현의 자동 학습**: 수동 특성 공학 없이 네트워크가 작업에 적합한 중간 표현을 자동으로 학습하는 능력을 입증했습니다.[1]

***

### 2. 논문이 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능

#### **문제 정의**

1986년 당시 신경망 연구의 근본적인 문제는 **깊은 다층 네트워크를 학습하는 방법의 부재**였습니다. 단순 퍼셉트론은 입출력이 직접 연결되었으나, 중간의 은닉층 뉴런들이 어떻게 활성화되어야 하는지 결정할 학습 규칙이 없었습니다. 이를 신용할당 문제(credit assignment problem)라고 합니다.[2][1]

#### **제안 방법: 역전파 알고리즘**

**순전파(Forward Pass)**:
네트워크는 입력층에서 출력층으로 신호를 전달합니다. 각 층 $l$의 유닛 $j$에 대한 총 입력은:

$$x_j = \sum_i w_{ij} y_i + b_j $$

여기서 $w_{ij}$는 가중치, $y_i$는 하층 유닛의 출력, $b_j$는 편향입니다.[1]

유닛의 출력은 비선형 함수(예: 시그모이드)를 적용합니다:

$$y_j = f(x_j) $$

**전체 오류 정의**:

$$E = \frac{1}{2}\sum_c \sum_j (y_{j,c} - d_{j,c})^2 $$

여기서 $c$는 입출력 쌍의 인덱스, $y$는 실제 출력, $d$는 목표 출력입니다.[1]

**역전파(Backward Pass)** - 핵심 알고리즘:

출력층에서 시작하여:

$$\frac{\partial E}{\partial y_j} = y_j - d_j $$

체인룰을 적용하여 각 가중치에 대한 편미분을 계산합니다:

$$\frac{\partial E}{\partial w_{ij}} = \frac{\partial E}{\partial x_j} \cdot \frac{\partial x_j}{\partial w_{ij}} = \frac{\partial E}{\partial x_j} \cdot y_i $$

은닉층으로 역전파할 때:

$$\frac{\partial E}{\partial y_i} = \sum_j \frac{\partial E}{\partial x_j} \cdot w_{ij} $$

이를 반복하여 모든 층에 적용합니다.[1]

**가중치 업데이트**:

기본적인 경사 하강법:

$$\Delta w(t) = -\varepsilon \frac{\partial E}{\partial w(t)} $$

가속 방법(momentum 포함):

$$\Delta w(t) = -\varepsilon \frac{\partial E}{\partial w(t)} + \alpha \Delta w(t-1) $$

여기서 $\varepsilon$는 학습률, $\alpha$는 모멘텀 계수입니다.[1]

#### **모델 구조**

논문에서 제시한 두 가지 주요 예제:

**1. 대칭성 감지 네트워크**
- 입력층: 8개 유닛 (1차원 이진 배열)
- 은닉층: 2개 유닛
- 출력층: 1개 유닛
- 작업: 입력 벡터가 중심 대칭인지 판단

학습 결과, 네트워크는 대칭성을 완벽히 감지하는 우아한 해를 찾았습니다. 각 은닉층 유닛은 중심점 주위로 거울 대칭인 가중치 패턴을 학습했으며, 비대칭 입력에 대해 은닉층이 활성화되어 출력을 억제했습니다.[1]

**2. 가족 관계 신경망 (5층 네트워크)**
- 입력층: 24개 유닛(인물) + 12개 유닛(관계) = 36개
- 2층: 각각 6개 유닛 (인물과 관계를 분산 표현으로 인코딩)
- 3층: 12개 유닛 (중앙 표현층)
- 4층: 6개 유닛 (미리앞층)
- 출력층: 24개 유닛 (결과 인물)

네트워크는 100개의 관계 트리플 중 4개를 제외한 100개로 학습되었으나, 학습하지 않은 4개의 트리플도 올바르게 예측했습니다. 특히 흥미로운 점은 영국 가족과 이탈리아 가족 간의 동형성(isomorphism)을 활용하여 구조를 공유했다는 것입니다.[1]

#### **성능 향상**

- **대칭성 감지**: 64개의 가능한 입력 패턴에서 완벽한 성능 (1,425번의 에포크 학습)[1]
- **가족 관계**: 학습 데이터에 포함되지 않은 패턴도 정확히 예측 (1500번의 에포크 학습)[1]
- 초기 무작위 가중치에서 시작하여 구조화된 내부 표현 자동 생성

***

### 3. 모델의 일반화 성능 향상 가능성 (중점)

#### **논문에서 다룬 일반화 메커니즘**

**내부 표현의 구조화**:
논문의 가족 관계 예제에서 가장 중요한 발견은 **은닉층이 작업 도메인의 기본 구조를 자동으로 캡처한다**는 점입니다. 예를 들어:[1]

- 유닛 1: 영국/이탈리아 국가 구분
- 유닛 2: 세대 인코딩
- 유닛 6: 가족 가지 구분

이러한 의미있는 특성들은 입출력 인코딩에는 명시되지 않았으나, 네트워크가 자동으로 학습했습니다. **이 구조화된 표현이 학습되지 않은 데이터로의 일반화를 가능하게 했습니다.**[1]

**차원성 증가의 이점**:
논문에서는 "국소 최소값에 빠지는 문제는 연결이 충분한 네트워크에서만 발생한다"고 지적했습니다. 연결을 몇 개 더 추가하면 가중치 공간에 추가 차원이 생겨 열악한 국소 최소값을 피할 수 있는 경로가 제공된다고 설명했습니다. 이는 현대의 **과적합(overparameterization)이 역설적으로 일반화를 개선한다**는 발견과 연결됩니다.[3][1]

#### **현대 연구 기반 일반화 향상 방법**

**1. 정규화 기법**

**Dropout**: 학습 중 무작위로 뉴런을 비활성화하여 이중 표현을 강제합니다. 이는 단일 뉴런에 과도하게 의존하는 것을 방지하고, 앙상블 효과를 생성합니다.[4]

**배치 정규화(Batch Normalization)**: 각 층의 활성화 분포를 정규화하여 내부 공변량 이동(internal covariate shift)을 완화합니다. 이를 통해:[4]
- 더 높은 학습률 사용 가능
- 훈련 가속화
- 일반화 성능 향상[4]

**가중치 감쇠(Weight Decay)**: 가중치의 크기를 제약하여 모델 복잡도를 억제합니다. 최근 연구에서는 가중치 감쇠가 암시적 정규화를 강화하여 일반화를 개선한다는 것을 입증했습니다.[5]

**2. 데이터 기반 접근**

**데이터 증강**: 명시적 정규화보다 효과적일 수 있으며, 네트워크가 학습할 수 있는 데이터의 양을 효과적으로 증가시킵니다.[6]

**Mixup**: 학습 샘플들을 선형 결합하여 새로운 훈련 예제를 생성합니다. 이는 더 평탄한 손실 경계(flat minima)로 수렴하게 하여 일반화를 개선합니다.[7]

**3. 아키텍처 기반 접근**

**ResNet(잔차 연결)**: 깊은 네트워크에서 경사 소실 문제를 해결하여 더 효과적인 일반화를 가능하게 합니다.[2]

**정규 직교 네트워크(Orthogonal DNNs)**: 가중치 행렬이 정규 직교 성질을 유지하도록 하여 생성 마진을 개선하고 일반화 오류의 이론적 상한을 감소시킵니다.[8]

#### **Rumelhart 논문과의 연결**

Rumelhart가 1986년에 보인 **"구조화된 내부 표현이 일반화를 개선한다"**는 발견은 현대 심층학습에서 다음과 같이 구현됩니다:[9]

- 낮은 층: 경계, 텍스처 같은 기본 특성 학습
- 중간 층: 복잡한 패턴과 관계 학습
- 높은 층: 추상적인 개념 학습

이 **계층적 특성 학습**이 일반화의 핵심입니다.[9]

***

### 4. 논문의 한계와 앞으로의 연구 고려사항

#### **논문에서 명시한 한계**

1. **국소 최소값 문제**: 경사 하강법은 전역 최소값을 찾도록 보장하지 않습니다. 그러나 논문은 "실제 대부분의 경우 열악한 국소 최소값에 빠지지 않는다"고 보고했습니다.[1]

2. **생물학적 타당성 부족**: "현재 형태의 학습 절차는 뇌의 학습의 생물학적 모델이 아닙니다." 이는 다음의 문제를 포함합니다:[1]
   - 정확한 대칭 가중치 전송 필요 (생물학적으로 비현실적)
   - 전역 오류 신호 필요 (국소 학습이 아님)

#### **현대 연구의 해결책과 새로운 과제**

**경사 소실/폭발 문제**:
1986년 논문에서 예상하지 못한 가장 큰 문제는 **경사 소실 문제(vanishing gradient problem)**입니다. 깊은 네트워크에서 역전파 시 경사가 지수적으로 감소하여 초기 층의 학습이 중단됩니다.[10][11][12]

현대 해결책:
- **ReLU 활성화 함수**: 시그모이드의 0-1 범위 문제 해결
- **배치 정규화**: 각 층의 활성화 분포 정규화[4]
- **ResNet**: 직결(skip connection) 통해 경사 직접 전달[2]
- **경사 클리핑**: 폭발하는 경사 제한[11]

**메모리 및 계산 효율성**:
역전파는 순전파의 모든 활성화를 저장해야 하므로 메모리 사용이 깊이에 정비례합니다. 최근 연구는 다음을 탐색 중입니다:[13]
- **예측 부호화(Predictive Coding)**: 역전파 대체 알고리즘[14][15]
- **뉴로모픽 구현**: 스파이킹 신경망에서의 역전파[16]

**생물학적 타당성 개선**:
최근 신경과학 기반 학습 알고리즘이 제안되었습니다:[17]
- 국소 학습 규칙 활용
- 대칭이 아닌 가중치 전송 구조
- 뇌와 유사한 피드백 메커니즘

***

### 5. 앞으로의 연구 시 고려할 점 (현대 최신 기반)

#### **1. 과적합과 일반화 간 균형**

최근 연구는 **"왜 큰 신경망이 과적합하지 않는가?"**라는 역설적 질문을 던집니다. 매개변수가 충분한 네트워크가 오히려 더 나은 일반화를 보이는 현상을 설명하는 이론이 발전 중입니다:[3]

- 암시적 정규화(implicit regularization)
- 손실 경계의 기하학적 성질
- SGD의 노이즈 효과

따라서 미래 연구는 **단순히 정규화를 추가하는 것보다 과적합 메커니즘 자체를 이해**하는 데 초점을 맞춰야 합니다.

#### **2. 다양한 작업에 맞춤형 아키텍처**

Rumelhart의 회귀적 네트워크 아이디어를 현대에 확장한 Transformer, RNN, CNN 등 다양한 아키텍처가 출현했습니다. 향후 연구는:[1]

- **작업 특정 표현**: 자연어 처리, 의료 영상, 강화학습 등 각 도메인에 최적화된 숨겨진 표현 구조
- **전이학습**: 대규모 사전학습 후 소규모 데이터에 미세조정하는 패러다임

#### **3. 해석 가능성**

Rumelhart 논문에서 보인 **의미있는 은닉층 표현 학습**이 현대 심층신경망에서도 가능한가? 이를 해석하고 제어하려는 연구:

- 특성 시각화 기법
- 어텐션 메커니즘의 해석
- 프로브(probing) 분석

#### **4. 효율성과 확장성**

역전파의 계산/메모리 비용이 모델 크기와 함께 증가하므로:[13]

- 그래프 신경망 최적화
- 분산 훈련의 효율성
- 뉴로모픽 하드웨어 구현

이들은 Rumelhart의 기본 원리를 유지하면서도 실제 대규모 애플리케이션에 확장 가능하게 합니다.

#### **5. 인과관계와 동적 표현**

단순 정적 표현을 넘어 **시간 변화하는 동적 표현** 학습이 중요해지고 있습니다:[18]
- 인과 발견(causal discovery)
- 반사실(counterfactual) 기반 학습
- 도메인 일반화(domain generalization)

***

## 결론

Rumelhart, Hinton, Williams의 1986년 논문은 **신용할당 문제 해결**과 **다층 네트워크의 자동 특성 학습**을 가능하게 함으로써 현대 심층학습의 기초를 마련했습니다. 특히 구조화된 내부 표현이 일반화를 개선한다는 통찰은 여전히 유효합니다.[2][1]

그러나 40년 가까운 시간이 경과하면서 다음이 명확해졌습니다:

- **경사 소실**: 해결되었으나 깊은 네트워크에서 여전히 고려 필요
- **규모와 일반화의 역설**: 이론적 설명이 진행 중
- **생물학적 타당성**: 여전히 미해결 과제
- **계산 효율성**: 현대 대규모 모델의 새로운 제약

향후 연구자들이 고려해야 할 가장 중요한 점은 **Rumelhart의 역전파가 신의 계산 도구임을 인정하면서도, 그 한계를 이해하고 이를 보완하는 새로운 패러다임을 개발**하는 것입니다.[15][14][17]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ad53d90e-0c27-47f7-a841-8682fb2a50bf/rumelhart1986.pdf)
[2](https://mbrenndoerfer.com/writing/history-backpropagation-deep-learning-training)
[3](https://arxiv.org/pdf/1611.03530.pdf)
[4](https://learnopencv.com/batch-normalization-and-dropout-as-regularizers/)
[5](https://proceedings.neurips.cc/paper_files/paper/2024/file/29496c942ed6e08ecc469f4521ebfff0-Paper-Conference.pdf)
[6](https://arxiv.org/pdf/1806.03852v3.pdf)
[7](https://arxiv.org/pdf/2308.03236.pdf)
[8](https://arxiv.org/pdf/1905.05929.pdf)
[9](https://deepai.org/machine-learning-glossary-and-terms/hidden-representation)
[10](https://arxiv.org/pdf/2303.09728v1.pdf)
[11](https://www.geeksforgeeks.org/deep-learning/vanishing-and-exploding-gradients-problems-in-deep-learning/)
[12](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
[13](https://arxiv.org/html/2509.19063v1)
[14](https://arxiv.org/pdf/2202.09467.pdf)
[15](https://arxiv.org/pdf/2304.02658.pdf)
[16](https://www.nature.com/articles/s41467-024-53827-9)
[17](https://arxiv.org/pdf/2308.07870.pdf)
[18](https://arxiv.org/html/2209.01610v3)
[19](https://arxiv.org/pdf/1404.7828.pdf)
[20](https://arxiv.org/pdf/1811.11987.pdf)
[21](https://drpress.org/ojs/index.php/ajst/article/download/16340/15863)
[22](https://arxiv.org/pdf/1808.02822.pdf)
[23](https://developer.nvidia.com/blog/deep-learning-nutshell-history-training/)
[24](https://www.digitalocean.com/community/tutorials/vanishing-gradient-problem)
[25](https://www.dataversity.net/articles/brief-history-deep-learning/)
[26](https://wandb.ai/mostafaibrahim17/ml-articles/reports/Decoding-Backpropagation-and-Its-Role-in-Neural-Network-Learning--Vmlldzo3MTY5MzM1)
[27](https://en.wikipedia.org/wiki/Backpropagation)
[28](https://arxiv.org/pdf/2402.02769.pdf)
[29](https://arxiv.org/pdf/1710.05468.pdf)
[30](https://arxiv.org/pdf/1301.3557.pdf)
[31](https://www.mdpi.com/1099-4300/26/1/7)
[32](https://arxiv.org/abs/2409.07606)
[33](https://nlp2024.tistory.com/131)
[34](https://questdb.com/glossary/hidden-layer-representations-in-deep-learning-for-finance/)
[35](https://dadev.tistory.com/entry/Intro-to-Deep-Learning-Dropout-and-Batch-Normalization)
[36](https://www.sciencedirect.com/science/article/abs/pii/S0098135419306878)
[37](https://www.nature.com/articles/s42005-024-01837-w)
