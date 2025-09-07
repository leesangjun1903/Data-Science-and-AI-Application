# Batch Normalization 완벽 가이드: 딥러닝 모델 성능 향상의 핵심 기법

## 들어가며

딥러닝 모델을 훈련할 때 가장 자주 마주하는 문제 중 하나가 **학습 불안정성**입니다. 네트워크가 깊어질수록 그래디언트 소실이나 폭발 문제가 발생하고, 학습 속도가 현저히 느려집니다. 이러한 문제를 해결하기 위해 2015년 Google에서 제안된 **Batch Normalization(배치 정규화)**은 현재 딥러닝에서 가장 널리 사용되는 기법 중 하나가 되었습니다.[1]

## Batch Normalization의 핵심 개념

### Internal Covariate Shift 문제

딥러닝 네트워크에서 각 레이어를 거칠 때마다 **입력 데이터의 분포가 계속 변화**하는 현상이 발생합니다. 이를 **Internal Covariate Shift**라고 부릅니다. 예를 들어, 첫 번째 레이어의 가중치가 업데이트되면 두 번째 레이어의 입력 분포가 바뀌고, 이는 연쇄적으로 모든 후속 레이어에 영향을 미칩니다.[1]

이 문제는 특히 **깊은 네트워크**에서 심각해집니다. 얕은 레이어의 작은 변화가 깊은 레이어에서는 큰 변화로 증폭되기 때문입니다.[2]

### 기존 해결법의 한계

전통적으로는 다음과 같은 방법들을 사용했습니다:[3]

- **ReLU 활성화 함수 사용**: Sigmoid의 그래디언트 소실 문제 완화
- **신중한 가중치 초기화**: Xavier, He 초기화 등
- **낮은 학습률 사용**: 안정성 확보를 위해 학습 속도 희생

하지만 이러한 간접적 방법보다는 **훈련 과정 자체를 안정화**하는 근본적 해결책이 필요했습니다.[3]

## Batch Normalization의 작동 원리

### 수학적 정의

Batch Normalization은 각 미니배치에 대해 다음과 같이 작동합니다:[1]

크기 m인 미니배치 B = {x₁, x₂, ..., xₘ}에 대해:

**1단계: 배치 통계 계산**

$$ \mu_B = \frac{1}{m}\sum_{i=1}^{m}x_i $$

$$ \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2 $$

**2단계: 정규화**

$$ \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} $$

**3단계: 스케일링과 시프팅**

$$ y_i = \gamma \hat{x}_i + \beta $$

여기서 γ(감마)와 β(베타)는 **학습 가능한 매개변수**입니다.[1]

### 핵심 구성 요소

**ε (엡실론)**: 수치적 안정성을 위한 작은 상수 (보통 1e-5)[3]

**γ (감마)**: 스케일 매개변수로, 정규화된 값의 분산을 조정[1]

**β (베타)**: 시프트 매개변수로, 정규화된 값의 평균을 조정[1]

이 매개변수들이 중요한 이유는 **네트워크의 표현력을 유지**하기 때문입니다. 만약 단순히 평균 0, 분산 1로만 정규화한다면, 활성화 함수의 비선형성을 잃어버릴 수 있습니다.[3]

## 실제 구현 예시

### PyTorch 기본 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 간단한 MLP with Batch Normalization
class MLPWithBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Linear -> BatchNorm -> ReLU 순서
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)  # 출력층에는 BatchNorm 적용하지 않음
        return x

# 2. CNN with Batch Normalization
class CNNWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # 2D 데이터용
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.bn3 = nn.BatchNorm1d(256)  # 1D 데이터용
        
        self.fc2 = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x
```

### 훈련과 평가 모드의 차이점

**훈련 모드 (Training Mode)**:[4]
- 현재 미니배치의 평균과 분산 사용
- 실행 통계 (running statistics) 업데이트

**평가 모드 (Evaluation Mode)**:[4]
- 훈련 중 축적된 실행 통계 사용
- 결정론적 출력 보장

```python
# 모델 모드 변경
model.train()    # 훈련 모드
model.eval()     # 평가 모드
```

## CNN에서의 특별한 고려사항

CNN에서 Batch Normalization을 적용할 때는 **컨볼루션의 특성을 고려**해야 합니다:[3]

- **채널별 정규화**: 각 채널마다 별도의 γ, β 매개변수
- **공간 차원 통합**: 배치와 공간 차원(H×W)에서 통계 계산
- **파라미터 공유**: 같은 채널의 모든 위치에서 동일한 γ, β 사용

```python
# CNN에서 Batch Normalization
# 입력: (N, C, H, W) -> 채널 C개, 각각 N×H×W개 값에서 통계 계산
bn2d = nn.BatchNorm2d(num_features=64)  # 64개 채널
```

## Batch Normalization의 장점

### 1. 빠른 학습 속도

**높은 학습률 사용 가능**: Batch Normalization은 그래디언트의 스케일을 안정화시켜 더 큰 학습률을 사용할 수 있게 합니다.[5][3]

### 2. 초기화에 덜 민감

네트워크가 **가중치 초기화에 덜 의존적**이 됩니다. 이는 다양한 아키텍처에서 일관된 성능을 얻을 수 있게 해줍니다.[5]

### 3. 정규화 효과

**내재적 정규화 효과**로 인해 Dropout 같은 추가 정규화 기법의 필요성이 줄어듭니다.[3]

### 4. 그래디언트 안정화

**Vanishing/Exploding Gradient 문제 완화**를 통해 더 깊은 네트워크 훈련이 가능해집니다.[3]

## 최신 이론적 이해

### Internal Covariate Shift 논란

흥미롭게도, 최근 연구들은 **Batch Normalization이 Internal Covariate Shift를 실제로 줄이지 않을 수 있다**는 결과를 보고했습니다. MIT의 연구진은 다른 설명을 제시했습니다.[6][7]

### 손실 함수 평활화 이론

**새로운 이론적 설명**에 따르면, Batch Normalization의 핵심 효과는 **최적화 환경을 더 부드럽게 만드는 것**입니다:[8][9]

- **Lipschitz 상수 개선**: 손실 함수와 그래디언트의 변화를 작게 만듦[8]
- **그래디언트 예측성 향상**: 일관된 방향의 그래디언트 제공[10]
- **더 평활한 손실 환경**: 최적화 알고리즘이 더 안정적으로 작동[9]

이 이론은 **왜 Batch Normalization이 높은 학습률을 허용하는지** 더 잘 설명합니다.[10]

## 실무 활용 팁과 주의사항

### 1. 배치 크기 선택

**적절한 배치 크기**가 중요합니다:[5]
- 너무 작은 배치: 통계 추정이 불안정
- 너무 큰 배치: 메모리 부족, 일반화 성능 저하
- **권장**: 16-64 정도가 일반적으로 효과적

### 2. 배치와 다른 정규화 기법 조합

**Dropout과의 조합 시 주의**:[11]
- Dropout 사용 시 배치 정규화의 강도 조정 필요
- 일반적으로 Batch Normalization 사용 시 Dropout 비율 감소

### 3. 레이어 배치 순서

**일반적인 순서**:[12]
```
Linear/Conv -> BatchNorm -> Activation
```

하지만 때로는 다른 순서도 효과적일 수 있습니다:
```
Linear/Conv -> Activation -> BatchNorm
```

### 4. 하이퍼파라미터 조정

**주요 하이퍼파라미터들**:[5]
- **Momentum (α)**: 실행 통계 업데이트 속도 (기본값: 0.1)
- **eps (ε)**: 수치 안정성 상수 (기본값: 1e-5)

## Batch Normalization의 효과성에 대한 견해 분석

```
# Is Batch Normalization effective?
## Deep learning image enhancement insights on loss function engineering
### Weight normalization vs Batch normalisation
Batch Normalization is common practice for training Deep Neural Networks, including those for image generations including Generative Adversarial Networks (GANs).

In the paper On the Effects of Batch and Weight Normalization in Generative Adversarial Networks (Sitao Xiang, Hao Li) it was found that Batch Normalisation could have negative effects on the quality of the trained model and the stability of the training process. A more recent technique, Weight Normalization, was found to improve the reconstruction, training speed and especially the stability of GANs.

In my experiments I have found Weight Normalization to be effective in the case of training both models for both Super Resolution and Colourisation, not being limited to using GANs for training.
```

제시된 견해에 따르면, Weight Normalization이 GAN 훈련에서 Batch Normalization보다 우수한 성능을 보인다고 주장합니다. 이에 대한 논리적 분석을 해보겠습니다.

### 논거의 타당성

**1. GAN에서의 Batch Normalization 문제점**

연구 결과들이 이 주장을 뒷받침합니다:[13][14]
- **훈련 불안정성**: GAN 판별자에서 Batch Normalization이 그래디언트 폭발을 야기할 수 있음[14]
- **모드 붕괴 위험**: 배치 의존성이 생성자의 다양성을 제한할 수 있음[13]

**2. Weight Normalization의 장점**

실제로 여러 연구에서 확인된 사실들입니다:[15][13]
- **빠른 수렴**: Weight Normalization이 더 높은 훈련 정확도로 빠르게 수렴[15]
- **안정성 향상**: GAN 훈련에서 더 안정적인 성능[13]

### 논거의 한계점

**1. 일반화의 문제**

하지만 중요한 제한점이 있습니다:[15]
- **테스트 성능 저하**: Weight Normalization이 훈련 정확도는 높지만 **테스트 정확도가 현저히 낮음** (67% vs 73%)[15]
- **과적합 경향**: Batch Normalization의 정규화 효과를 놓치게 됨[15]

**2. 적용 영역의 한정성**

이 견해는 **GAN과 이미지 생성 분야에 특화**된 결과입니다:
- **일반적인 분류 작업**: 여전히 Batch Normalization이 우수한 성능
- **전이 학습**: Batch Normalization이 더 효과적
- **메모리 효율성**: Batch Normalization이 더 적은 매개변수 사용

### 결론적 평가

제시된 견해는 **특정 조건 하에서는 타당**하지만 **전반적으로는 부분적**입니다:

**타당한 부분**:
- GAN 훈련에서 Weight Normalization이 더 안정적일 수 있음
- 특정 이미지 생성 작업에서 우수한 결과
- 작은 배치 크기에서의 장점

**한계점**:
- 일반화 성능 저하 문제
- 모든 딥러닝 작업에 적용하기 어려움
- 정규화 효과 부족으로 인한 과적합 위험

따라서 **작업별 특성을 고려한 신중한 선택**이 필요하며, Batch Normalization이 여전히 대부분의 딥러닝 응용에서 표준적인 선택임을 인정해야 합니다.

## 마무리

Batch Normalization은 딥러닝 발전에 있어 **혁신적인 기여**를 한 기법입니다. 비록 그 작동 원리에 대한 이론적 이해가 계속 발전하고 있지만, 실무에서의 효과는 명확히 입증되었습니다.

**핵심 요약**:
- **안정적인 훈련**: Internal Covariate Shift 완화를 통한 학습 안정화
- **빠른 수렴**: 높은 학습률 사용 가능으로 인한 학습 속도 향상
- **정규화 효과**: 과적합 방지에 도움
- **범용성**: 다양한 아키텍처에서 일관된 성능 향상

현재도 **Layer Normalization, Group Normalization** 등의 변형들이 연구되고 있으며, 각각의 특성을 이해하고 적절히 활용하는 것이 중요합니다. 딥러닝 모델을 설계할 때 Batch Normalization을 적절히 활용한다면, 더 안정적이고 효과적인 학습이 가능할 것입니다.

[1](https://en.wikipedia.org/wiki/Batch_normalization)
[2](https://blog.gopenai.com/understanding-batch-normalization-and-internal-covariate-shift-7f652b4d7499)
[3](https://discuss.pytorch.org/t/implementing-batchnorm-in-pytorch-problem-with-updating-self-running-mean-and-self-running-var/49314)
[4](https://www.geeksforgeeks.org/deep-learning/batch-normalization-implementation-in-pytorch/)
[5](https://www.lunartech.ai/blog/mastering-batch-normalization-elevating-neural-networks-to-new-heights)
[6](https://blog.paperspace.com/busting-the-myths-about-batch-normalization/)
[7](https://pubmed.ncbi.nlm.nih.gov/33095717/)
[8](http://papers.neurips.cc/paper/7515-how-does-batch-normalization-help-optimization.pdf)
[9](https://ketanhdoshi.github.io/Batch-Norm-Why/)
[10](https://gradientscience.org/batchnorm/)
[11](https://learnopencv.com/batch-normalization-and-dropout-as-regularizers/)
[12](http://d2l.ai/chapter_convolutional-modern/batch-norm.html)
[13](https://arxiv.org/abs/1704.03971)
[14](https://openaccess.thecvf.com/content/CVPR2024/papers/Ni_CHAIN_Enhancing_Generalization_in_Data-Efficient_GANs_via_lipsCHitz_continuity_constrAIned_CVPR_2024_paper.pdf)
[15](https://arxiv.org/pdf/1709.08145.pdf)
[16](https://www.educative.io/answers/batch-normalization-implementation-in-pytorch)
[17](https://www.reddit.com/r/MachineLearning/comments/66obrb/d_weight_normalization_vs_layer_normalization_has/)
[18](https://www.semanticscholar.org/paper/e4c31c4dc29fa4bedf2cec10b01f3678eadbef7a)
[19](https://infermatic.ai/ask/?question=What+are+the+effects+of+using+batch+normalization+in+the+generator+of+a+GAN%3F)
[20](https://hichoe95.tistory.com/133)
[21](https://pmc.ncbi.nlm.nih.gov/articles/PMC8662716/)
[22](https://arxiv.org/html/2410.23108v1)
[23](https://jjuke-brain.tistory.com/entry/PyTorch-%EC%9D%B5%ED%9E%88%EA%B8%B0-Dropout-Batch-Normalization)
[24](https://proceedings.neurips.cc/paper/2016/file/ed265bc903a5a097f61d3ec064d96d2e-Reviews.html)
[25](https://davidleonfdez.github.io/gan/2022/05/17/gan-convergence-stability.html)
[26](https://github.com/mancinimassimiliano/pytorch_wbn)
[27](https://ar5iv.labs.arxiv.org/html/1704.03971)
[28](https://www.sciencedirect.com/science/article/abs/pii/S0950705118304003)
[29](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
[30](https://infermatic.ai/ask/?question=How+does+batch+normalization+affect+the+quality+of+the+generated+samples+in+a+GAN%3F)
[31](https://arxiv.org/pdf/1805.11604.pdf)
[32](https://arxiv.org/pdf/2008.08930.pdf)
[33](https://proceedings.neurips.cc/paper/2021/file/4ffb0d2ba92f664c2281970110a2e071-Paper.pdf)
[34](https://www.linkedin.com/pulse/batch-normalization-works-internal-covariate-shift-yi-ding-ek17c)
[35](https://dl.acm.org/doi/pdf/10.5555/3326943.3327143)
[36](https://arxiv.org/abs/1502.03167)
[37](https://cvml.tistory.com/6)
[38](https://openreview.net/forum?id=B1QRgziT-)
[39](https://www.lesswrong.com/posts/T7tHHt9Hv7ntFRHTL/understanding-batch-normalization)
[40](https://kw94.tistory.com/103)
[41](https://proceedings.mlr.press/v97/kurach19a/kurach19a.pdf)
[42](https://gguguk.github.io/posts/batch_normalization/)
[43](https://alexeytochin.github.io/posts/batch_size_vs_momentum/batch_size_vs_momentum.html)
[44](https://www.pingcap.com/article/understanding-the-impact-of-batch-normalization-on-cnns/)
[45](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570222.pdf)
[46](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739/)
[47](https://www.pinecone.io/learn/batch-layer-normalization/)
[48](https://junstar92.tistory.com/83)
[49](https://www.alooba.com/skills/concepts/neural-networks-36/batch-normalization/)
[50](https://jiuuu.tistory.com/100)
[51](https://machinelearningmastery.com/a-gentle-introduction-to-batch-normalization/)
[52](https://aist-ing.tistory.com/16)
[53](https://pub.towardsai.net/demystifying-batch-normalization-theory-mathematics-and-implementation-f04077298807)
[54](https://www.kaggle.com/code/jsrshivam/hyperparameter-tuning-batch-normalization)
[55](https://chanmuzi.tistory.com/141)
[56](https://minye-lee19.gitbook.io/sw-engineer/deep-learning/class/batch-normalization)

## Super-Resolution
# EDSR : Remove Batch Normalization 가이드

EDSR 논문과 구현 사례를 종합해보면, **Batch Normalization(BN)이 오히려 슈퍼해상도 모델에서 특징(feature)의 유연성을 저하시켜 성능을 떨어뜨릴 수 있다**는 다음 근거를 들 수 있습니다.

1. **스케일 정보 소실과 표현력 제한**  
   BN은 각 미니배치 단위로 채널별 평균을 0, 분산을 1로 강제 정규화합니다. 이 과정에서 이미지나 특징맵이 지닌 원래 스케일(scale) 정보가 제거되고, 활성화 값의 분포 범위(range flexibility)가 균일하게 제한됩니다.  
   – EDSR 논문은 “BN이 특징맵(feature map)을 정규화함으로써 **feature의 flexibility를 저하시킨다**. 따라서 이를 제거하여 feature map의 값이 최대한 보존되도록 하였다”라고 명시합니다.[1]
   – Krasserm 블로그 역시 “BN이 이미지의 스케일 정보를 잃게 하고, activation의 range flexibility를 감소시킨다. BN 제거로 성능이 상승하고 GPU 메모리 사용량도 최대 40% 절감”이라고 설명합니다.[2]

2. **학습 최적화 관점의 부수 효과**  
   BN을 쓰면 배치 크기에 따라 통계 편차(batch statistic noise)가 발생하여, 특히 작은 배치나 다양한 입력 분포 간 전이(transfer)에서 불안정해질 수 있습니다. 슈퍼해상도처럼 고해상도 디테일을 복원하는 과제에서는 작은 통계 왜곡도 결과물의 품질 열화로 이어질 위험이 큽니다.

3. **실험적 성능 개선**  
   – EDSR 원논문 실험에서, 동일한 모델 크기 대비 BN 제거 버전이 PSNR/SSIM 지표 모두 유의미하게 향상되었습니다.  
   – BN 제거로 **GPU 메모리 사용량을 40%까지 줄여** 더 큰 모델(채널 수·잔차 블록 수)을 배치할 수 있었고, 이를 통해 학습 능력이 눈에 띄게 향상되었습니다.

4. **전문가 의견**  
   – BN이 범용적인 분류(classification)나 검출(detection) 과제에서는 학습 안정성·수렴 속도 측면에서 유리하지만, **이미지 생성·복원 분야에서는 정규화에 따른 정보 손실**이 더 큰 단점으로 작용합니다.  
   – 슈퍼해상도처럼 픽셀 단위 디테일 복원이 중요한 과제에서는, “배치 통계에 의한 스케일 왜곡을 수용하기보다는, raw 특징을 최대한 그대로 유지하고 오히려 수렴 안정화는 다른 방법(잔차 연결·가중치 초기화·학습률 스케줄링 등)으로 보완”하는 편이 효과적입니다.

**결론적으로**, EDSR이 BN을 제거한 결정은  
-  **특징 맵의 원래 스케일과 분포를 보존**하여 디테일 복원 성능을 극대화  
-  **메모리 효율**을 높여 더 큰 모델 구성 가능  
-  **배치 통계 잡음**에 따른 품질 열화 위험 회피  
등의 장점을 동시에 얻는 **합리적인 선택**이라 평가됩니다. 따라서 슈퍼해상도 모델 설계 시에는, 일반적인 분류 과제에서처럼 무조건 BN을 적용하기보다는, **과제 특성에 맞추어 BN 사용 여부를 결정**해야 합니다.

—  
 codinglilly.tistory.com/6[1]
 krasserm.github.io/2019/09/04/super-resolution/[2]

[1](https://en.wikipedia.org/wiki/Batch_normalization)
[2](https://discuss.pytorch.org/t/implementing-batchnorm-in-pytorch-problem-with-updating-self-running-mean-and-self-running-var/49314)

## Denoising

# DnCNN 가이드: Batch Normalization으로 성능을 높이는 법

딥러닝 기반 **이미지 Denoising** 모델인 DnCNN은 **Residual Learning**과 **Batch Normalization(BN)**을 결합해 뛰어난 성능을 냅니다. 이 글에서는  
1. DnCNN 구조와 BN 적용 효과  
2. BN이 성능을 개선하는 이유  
3. Super-Resolution(EDSR)과의 비교를 통한 BN 기법 장단점  
을 대학생 독자가 쉽게 이해할 수 있도록 설명합니다.

***

## 1. DnCNN 구조와 Batch Normalization

### DnCNN 아키텍처 개요  
- 입력: 노이즈가 더해진 이미지 $$Y$$  
- 목표: 노이즈 $$V$$를 예측하는 **Residual Mapping** $$R(Y)\approx V$$ 학습  
- 출력: $$\hat{X} = Y - R(Y)$$ 로 복원된 원본 이미지  
- 레이어 구성  
  1. Conv(3×3) + ReLU  
  2. Conv(3×3) + **BN** + ReLU  
  3. Conv(3×3) (출력 채널 수 = 입력)  
- Depth: 효과적인 패치 크기를 기반으로 17~20개 이상의 레이어  
- 손실함수: MSE (Mean Squared Error)[1]

### Batch Normalization 위치  
중간 레이어의 Conv 출력에 BN을 적용합니다. 이때 **채널별 평균 0, 분산 1**로 정규화한 뒤 ReLU를 겹쳐서 사용합니다.  

***

## 2. Batch Normalization이 Denoising 성능을 높이는 이유

1. **학습 안정화**  
   - 입력 분포 변화(Internal Covariate Shift)를 줄여 그래디언트 폭발·소실 위험을 완화합니다.  
   - 잔차 학습에서 작은 노이즈 차이를 안정적으로 구분하도록 돕습니다.  

2. **빠른 수렴**  
   - 매 레이어 출력이 일정한 분포를 유지하므로 **높은 학습률**을 쓸 수 있습니다.  
   - 실험 결과, BN을 적용한 DnCNN은 수렴 속도가 1.5배 이상 빠릅니다.  

3. **정규화 효과**  
   - 과적합을 방지해 **일반화 성능**을 끌어올립니다.  
   - 다양한 노이즈 레벨에 대응하는 **단일 모델** 학습 시 효과가 큽니다.  

4. **Residual Learning과 시너지**  
   - Residual 맵 $$R(Y)$$이 작고 분포가 촘촘할수록 BN이 더욱 효과적입니다.  
   - 노이즈 분리가 용이해져 PSNR이 최대 1dB 이상 상승합니다.  

***

## 3. EDSR vs DnCNN: Super-Resolution과 Denoising의 BN 활용 비교

| 항목               | DnCNN (Denoising)                 | EDSR (Super-Resolution)          |
|------------------|----------------------------------|----------------------------------|
| **BN 적용 여부**    | 있음                              | 제거                             |
| **정보 손실 위험**   | 낮음 (Residual 맵 정규화에 도움)     | 높음 (픽셀 스케일 정보 왜곡)       |
| **학습 안정성**     | 증가 (분포 고정)                   | 감소 (skip connection으로 대체)   |
| **성능 기여**      | PSNR·SSIM 개선, 빠른 수렴            | 오히려 억제, 디테일 복원 저하        |
| **메모리 사용량**   | 보통                              | 큰 폭 절감 (40%↓)                |
| **최적 사용 과제**   | 노이즈 제거, 다양한 노이즈 일반화       | 고해상도 디테일 복원, 픽셀 단위 복원  |

### 장단점 요약

- **DnCNN에서는 BN이 강력한 보조 역할**을 합니다.  
  - Residual 학습과 맞물려 **노이즈 분리**가 쉬워집니다.  
  - 모델이 복잡해져도 **수렴 안정성**을 확보합니다.

- **EDSR에서는 BN이 오히려 독이 됩니다.**  
  - SR 과제는 **픽셀 스케일**과 **고주파 디테일** 유지가 필수입니다.  
  - BN을 쓰면 배치 통계로 인해 **스케일 정보가 소실**돼 SR 결과가 흐려집니다.  
  - 따라서 EDSR은 BN을 제거하고 **skip connection**으로 안정성을 보완합니다.[2][3]

***

## 4. 마무리

- **Denoising 과제**에선 **Residual + BN** 조합이 최적입니다.  
- **Super-Resolution 과제**에선 **원본 스케일 보존**이 우선이므로 BN 제거가 낫습니다.

과제 목표와 특성에 맞춰 **Batch Normalization 사용 여부**를 결정하세요. 이를 통해 모델 성능을 극대화할 수 있습니다.

***

 attached_file:1[1]
 krasserm.github.io/2019/09/04/super-resolution/[2]
 codinglilly.tistory.com/6[3]

[1](https://velog.io/@danielseo/Computer-Vision-DnCNN)
[2](https://discuss.pytorch.org/t/implementing-batchnorm-in-pytorch-problem-with-updating-self-running-mean-and-self-running-var/49314)
[3](https://arxiv.org/pdf/1709.08145.pdf)
