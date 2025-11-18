# Optimizing Millions of Hyperparameters by Implicit Differentiation

### 1단계: 핵심 주장과 주요 기여 요약

**핵심 주장**[1]

이 논문의 주요 주장은 **격함수 정리(Implicit Function Theorem, IFT)**와 **효율적인 역 헤시안 근사(inverse Hessian approximation)**를 결합하여 수백만 개의 하이퍼파라미터를 최적화할 수 있다는 것입니다. 기존 방법들은 저차원 하이퍼파라미터 최적화(random search, Bayesian optimization)에는 효과적이지만, 고차원 영역에서는 확장성이 떨어집니다. 저자들은 검증 손실에 대한 하이퍼파라미터 기울기(hypergradient)를 효율적으로 계산하는 방법을 제시함으로써 이 문제를 해결합니다.

**주요 기여**[1]

- **안정적인 역 헤시안 근사**: 상수 메모리 비용으로 작동하는 Neumann 급수 기반 근사 제안
- **IFT와 최적화 미분화 관계**: IFT가 제한된 단계의 미분화의 극한이라는 이론적 결과 제시
- **대규모 신경망 적용**: AlexNet 및 LSTM 기반 언어 모델에 적용 가능
- **다양한 고차원 하이퍼파라미터 응용**: 데이터 증강 네트워크, 데이터 증류, 개별 정규화 파라미터 튜닝 등
- **훈련-검증 분할 지침**: 많은 하이퍼파라미터 튜닝 시 적절한 데이터 분할 비율 연구

***

### 2단계: 문제 정의, 제안 방법 및 모델 구조

**해결하는 문제**[1]

신경망의 일반화 성능은 하이퍼파라미터 선택에 크게 의존합니다. 그러나 두 가지 핵심 문제가 존재합니다:

1. **고차원 하이퍼파라미터 최적화의 확장성**: 정규화 기법이 수십 개이고 각각 여러 하이퍼파라미터를 가질 때, 그리고 데이터 증강이나 데이터 증류처럼 가중치만큼 많은 하이퍼파라미터가 필요한 경우 기존 방법은 계산적으로 실행 불가능합니다.

2. **"Pure-Response" 게임의 독특한 어려움**: 하이퍼파라미터가 훈련 손실에 직접 영향을 주지 않고 오직 최적화된 가중치를 통해서만 검증 손실에 영향을 줄 때, 직접 기울기가 0이 되어 간접 기울기를 계산해야 합니다.

**제안하는 방법: 중첩 최적화 프레임워크**[1]

논문은 다음 중첩 최적화 문제를 풀어야 합니다:

```math
\lambda^* := \arg\min_{\lambda} L^*_V(\lambda) \text{ where } L^*_V(\lambda) := L_V(\lambda, w^*(\lambda)) \text{ and } w^*(\lambda) := \arg\min_w L_T(\lambda, w)
```

여기서 $$\lambda$$는 하이퍼파라미터, $$w$$는 신경망 가중치, $$L_T$$는 훈련 손실, $$L_V$$는 검증 손실입니다.

**하이퍼기울기(Hypergradient) 계산**[1]

하이퍼기울기는 두 개의 항으로 분해됩니다:

```math
\frac{\partial L^*_V(\lambda)}{\partial \lambda} = \underbrace{\frac{\partial L_V}{\partial \lambda}}_{\text{직접 기울기}} + \underbrace{\frac{\partial L_V}{\partial w} \cdot \frac{\partial w^*}{\partial \lambda}}_{\text{간접 기울기}}
```

하이퍼파라미터 최적화에서는 직접 기울기가 0이 므로 간접 기울기를 계산해야 합니다.

**암시 함수 정리(IFT) 활용**[1]

최적 가중치 $$w^\*(\lambda)$$ 가 고정점($$\frac{\partial L_T}{\partial w}|_{\lambda, w^*(\lambda)} = 0$$)에서 IFT를 적용하면:

$$
\frac{\partial w^*}{\partial \lambda} = -\left(\frac{\partial^2 L_T}{\partial w \partial w^T}\right)^{-1} \frac{\partial^2 L_T}{\partial w \partial \lambda^T}
$$

핵심은 훈련 헤시안의 역을 계산하는 것인데, 이는 현대 신경망에서 $$O(m^3)$$ 복잡도로 인해 실행 불가능합니다.

**Neumann 급수를 통한 역 헤시안 근사**[1]

이를 해결하기 위해 Neumann 급수 근사를 사용합니다:

$$
\left(\frac{\partial^2 L_T}{\partial w \partial w^T}\right)^{-1} \approx \sum_{j=0}^{i-1} \left(I - \frac{\partial^2 L_T}{\partial w \partial w^T}\right)^j
$$

**Algorithm 1: 하이퍼파라미터 최적화**[1]

```
Algorithm 1: 기본 루프
1. 초기화: 하이퍼파라미터 λ', 가중치 w'
2. 수렴할 때까지:
   - k번 반복: w' ← w' - α·∂L_T/∂w|λ',w'
   - λ' ← λ' - hypergradient(L_V, L_T, λ', w')
```

**Algorithm 2: 하이퍼기울기 계산**[1]

```
Algorithm 2: hypergradient 계산
1. v1 = ∂L_V/∂w|λ',w'
2. v2 = approxInverseHVP(v1, ∂L_T/∂w)  # 역 헤시안 벡터곱
3. v3 = grad(∂L_T/∂λ, w, grad_outputs=v2)
4. return ∂L_V/∂λ|λ',w' - v3
```

**Algorithm 3: Neumann 근사**[1]

```
Algorithm 3: approxInverseHVP (Neumann 근사)
1. 초기화: p = v
2. i번 반복:
   - v ← v - α·grad(∂L_T/∂w, w, grad_outputs=v)
   - p ← p - v
3. return p  # (∂L_T/∂w∂w^T)^(-1)·v의 근사
```

**최적 가중치로부터의 펼친 최적화**[1]

펼친 SGD 최적화의 재귀식은 다음과 같습니다:

$$
w_{i+1}(\lambda)=T(\lambda,w_i)=w_i(\lambda)-\alpha\frac{\partial L_T(\lambda,w_i(\lambda))}{\partial w}
$$

이로부터 도출되는 최적 가중치에서의 기울기 변화율은:

$$
\frac{\partial w_{i+1}}{\partial \lambda}=-\sum_{j\leq i}\left(\prod_{k<j}\left(I-\frac{\partial^2 L_T}{\partial w\partial w^T}\right)_{\lambda,w_{i-k}(\lambda)}\right)\frac{\partial^2 L_T}{\partial w\partial \lambda^T}\bigg|_{\lambda,w_{i-j}(\lambda)}
$$

***

### 2.1: 이론적 핵심 결과

**Theorem 2 (Neumann-SGD)**[1]

주어진 SGD 최적화 재귀에서, $$w_0 = w^*(\lambda)$$이고 $$I + \frac{\partial^2 L_T}{\partial w\partial w^T}$$가 축약적(contractive)이면:

```math
\frac{\partial w_{i+1}}{\partial \lambda} = -\left(\sum_{j < i}\left(I - \frac{\partial^2 L_T}{\partial w\partial w^T}\right)^j\right)\frac{\partial^2 L_T}{\partial w\partial \lambda^T}\bigg|_{w^*(\lambda)}
```

그리고 극한에서:

$$
\lim_{i \to \infty}\frac{\partial w_{i+1}}{\partial \lambda} = -\left(\frac{\partial^2 L_T}{\partial w\partial w^T}\right)^{-1}\frac{\partial^2 L_T}{\partial w\partial \lambda^T}\bigg|_{w^*(\lambda)}
$$

이는 **무한 단계의 펼친 미분화가 IFT 기반 계산과 수렴함**을 의미합니다.

**최종 하이퍼기울도 표현**[1]

$$
\frac{\partial L^*_V(\lambda)}{\partial \lambda} = \frac{\partial L_V}{\partial \lambda}\bigg|_{\lambda',w'} + \frac{\partial L_V}{\partial w}\bigg|_{\lambda',w'} \times \left(-\left(\frac{\partial^2 L_T}{\partial w\partial w^T}\right)^{-1}\bigg|_{\lambda',w'} \times \frac{\partial^2 L_T}{\partial w\partial \lambda^T}\bigg|_{\lambda',w'}\right)
$$

***

### 3단계: 성능 향상 및 한계

**성능 향상**[1]

논문의 실험 결과는 광범위한 응용에서 효과를 입증합니다:

**1. 데이터셋 증류 (Dataset Distillation)**[1]
- MNIST: 7,840개 하이퍼파라미터 (28×28×10 픽셀)
- CIFAR-10: 30,720개 하이퍼파라미터
- CIFAR-100: 300,720개 하이퍼파라미터
- 인식 가능한 증류 이미지 생성 가능

**2. 학습된 데이터 증강 네트워크**[1]
- 6,659개 하이퍼파라미터 (U-Net 기반 증강 네트워크)
- ResNet18 on CIFAR-10 결과:
  - 검증 정확도: 95.1% (3 Neumann 항)
  - 테스트 정확도: 94.6%
  - 정체 초기화 대비 +2.5% 개선

**3. LSTM 언어 모델 (Penn TreeBank)**[1]
- 1,691,951개 개별 정규화 하이퍼파라미터 튜닝
- 검증 혼란도(Perplexity): 68.18 (많은 하이퍼파라미터 튜닝)
- 기존 Self-Tuning Networks (STN): 70.30
- 계산 시간: 18.5k초 (STN 25k초 대비 26% 단축)

**4. 검증 세트 과적합 능력**[1]
- 50개 훈련/검증 샘플 설정에서 AlexNet으로 100% 검증 정확도 달성
- 테스트 정확도는 현저히 낮음 (과적합 확인)

**메모리 효율성 비교**[1]

| 방법 | 메모리 복잡도 |
|------|--------------|
| 최적화 미분화 (i단계 펼침) | $$O(PI + H)$$ |
| Linear Hypernetwork | $$O(PH)$$ |
| Self-Tuning Networks | $$O((P+H)K)$$ |
| **Neumann/CG IFT (제안)** | **$$O(P + H)$$** |

여기서 $$P$$는 가중치 수, $$H$$는 하이퍼파라미터 수, $$I$$는 펼침 단계 수, $$K$$는 병목 크기입니다.

---

### 3.1: 일반화 성능 향상 가능성 (중점)

**일반화 성능 개선 메커니즘**[1]

논문의 한 가지 중요한 발견은 훈련-검증 분할 비율이 하이퍼파라미터 수에 따라 달라야 한다는 것입니다:

**Figure 9: 분할 비율 영향**[1]
- **소수 하이퍼파라미터** (전역 가중치 감소, 1개):
  - 최적 검증 비율: ~10% (표준 관행 일치)
  - 재훈련 여부에 따른 성능 차이: 미미

- **많은 하이퍼파라미터** (개별 가중치당 감소):
  - 최적 검증 비율: 25-50% (훈련 전 최적화)
  - 재훈련 후 최적 검증 비율: 변화 (overfitting 위험 증가)

**일반화 성능 향상의 실증적 증거**[1]

1. **데이터 증강 학습**:
   - U-Net 기반 증강 네트워크는 검증 손실을 줄임
   - 테스트 정확도: 94.6% (전체 모집단에서의 일반화 개선)
   - 여러 무작위 시작 간 분산 감소: 0.002 (더 안정적인 일반화)

2. **LSTM 정규화 최적화**:
   - 개별 드롭아웃/DropConnect 초매개변수 튜닝: 68.18 perplexity
   - No HO: 75.72 perplexity
   - 개선폭: 10% 이상

3. **역 헤시안 근사 비교**:[1]
   - Neumann 근사는 더 많은 항을 사용할수록 코사인 유사성 개선
   - 5-Neumann이 Conjugate Gradient와 비슷하거나 우수한 방향성 정확도

**일반화 개선의 이론적 기초**[1]

Neumann-IFT 근사를 통한 일반화 개선은 두 가지 메커니즘을 통합합니다:

1. **정규화 효과**: 많은 하이퍼파라미터를 동시에 튜닝할 때, 충분한 검증 데이터를 사용하면 과적합을 방지합니다.

2. **역 헤시안 정보 활용**: 헤시안의 곡률 정보(2차 정보)를 사용하여 가중치 공간의 기하학적 특성을 고려한 하이퍼파라미터 조정이 가능합니다. 이는 1차 기울기만 사용하는 방법보다 더 나은 최적점을 찾습니다.

**근사 오차 분석**[1]

IFT를 정확히 적용하려면 $$\frac{\partial L_T}{\partial w}\bigg|_{\lambda',w'} = 0$$이어야 하지만, 실제로는 근사 해 $$\tilde{w}^*(\lambda)$$를 사용합니다. Neumann 근사의 오차는:

$$
\left|\left(\frac{\partial^2 L_T}{\partial w\partial w^T}\right)^{-1} - \sum_{j=0}^{i-1}\left(I - \frac{\partial^2 L_T}{\partial w\partial w^T}\right)^j\right|
$$

이는 헤시안의 조건 수(condition number)에 따라 수렴 속도가 결정됩니다.

***

### 4단계: 한계

**IFT 적용 조건**[1]

1. **미분가능성**: $$L_V : \Lambda \times W \to \mathbb{R}$$은 미분가능하고, $$L_T : \Lambda \times W \to \mathbb{R}$$은 이중 미분가능해야 함

2. **헤시안 가역성**: $$\frac{\partial^2 L_T}{\partial w \partial w^T}\bigg|_{w^*(\lambda)}$$가 가역이어야 함

3. **최적점의 미분가능성**: $$w^*: \Lambda \to W$$가 미분가능해야 함

**이산 하이퍼파라미터**[1]
- 연속 하이퍼파라미터만 직접 최적화 가능
- 이산 선택(예: 숨겨진 유닛 수)은 연속 이완 필요

**최적화기 하이퍼파라미터**[1]
- 손실 다양체를 변화시키지 않는 하이퍼파라미터(예: 최적화기 선택)는 직접 적용 불가

**헤시안 조건 수 의존성**[1]
- Neumann 근사의 수렴은 $$I - \frac{\partial^2 L_T}{\partial w \partial w^T}$$가 축약일 것 요구
- 조건 수가 높은 경우 더 많은 항 필요

***

### 5단계: 이후 연구에 미치는 영향

**현재 연구 트렌드와 영향 (2020-2025)**[2][3][4][5]

이 논문은 다음 분야의 발전을 촉발했습니다:

**1. 암시 미분화 확장 연구**[3][4][5]

- **Non-smooth 최적화로 확장**: Bertrand et al. (2022)은 내부 문제가 convex하지만 non-smooth일 때의 암시 미분화 확장
  
- **Nystrom 방법**: 2023년 연구에서는 Nystrom 방법을 사용하여 역 헤시안 벡터곱을 더 효율적으로 계산
  - 반복 근사보다 빠름
  - 대규모 하이퍼파라미터 최적화 및 메타 학습에서 우수한 성능

- **Koopman 연산자를 통한 하이퍼기울도**: 2024년 Hataya & Kawahara는 전역 하이퍼기울도의 신뢰성과 국지 하이퍼기울도의 효율성을 결합

**2. 양수준 최적화의 이론적 진전**[6][7]

- **수렴 분석**: Ji et al. (2020-2021)은 AID (Approximate Implicit Differentiation)와 ITD (Iterative Differentiation) 기반 양수준 알고리즘의 포괄적 수렴 분석

- **하한 결과**: 양수준 최적화의 첫 번째 하한 제시, 특정 조건에서 최적성 증명

**3. AutoML 및 메타학습 응용 확대**[8][9]

- **Bilevel 최적화 프레임워크**: AutoML 방법론으로서의 중요성 증가
  - 신경망 아키텍처 탐색 (NAS)
  - 메타 학습 (Meta-Learning)
  - 데이터 증강 자동화

- **함수형 양수준 최적화**: 2024 NeurIPS에서는 함수 공간 관점의 양수준 최적화 제시
  - 과매개변수화된 신경망에 적용 가능
  - 기존 매개변수 설정의 강 convexity 가정 불필요

**4. 분산 감소 및 확률론적 개선**[10]

- 2024-2025년 연구: 확률론적 하이퍼기울도 추정의 분산 감소
  - 비동기 Krasnosel'ski-Mann 반복을 통한 거의 확실한 수렴
  - 기존 방법 대비 더 낮은 점근 분산

**5. 일반화 성능에 대한 이론적 이해**[11][12][13][14][15][16]

- **암시 정규화 메커니즘**: 2017-2022년 연구들이 암시 정규화와 일반화의 관계 규명

- **Initialization의 중요성**: 2022년 연구에서 초기화가 신경망 일반화 성능의 핵심 요소임 입증

- **전체 배치 학습과 정규화**: 확률 경사하강법의 암시 정규화를 명시 정규화로 완전히 대체 가능

***

### 6단계: 앞으로의 연구 시 고려할 점

**1. 이론적 과제**

- **Non-convex 내부 문제의 수렴 분석**: 현재 대부분 strongly-convex 내부 문제에 대한 이론만 존재하며, 신경망의 비볼록 특성을 완전히 포착하는 이론 부족

- **오버파라미터화 역할**: 오버파라미터화된 신경망에서 하이퍼파라미터 최적화의 영향을 체계적으로 분석할 필요

- **불연속 하이퍼파라미터 처리**: 연속 이완 방법의 오류 한계와 수렴성에 대한 이론적 분석 부족

**2. 알고리즘 개선**

- **단일 루프 알고리즘**: 계산 복잡도를 더 줄일 수 있는 단일 루프 양수준 최적화 방법 개발

- **두 번째 순서 정보의 효율적 활용**: Kronecker-factored approximate curvature (KFAC) 같은 구조화된 Hessian 근사와의 결합

- **분산 감소**: 확률론적 설정에서 하이퍼기울도 추정의 분산을 더욱 감소시킬 수 있는 기술

**3. 실제 응용 과제**

- **계산 자원 최적화**: 매우 큰 모델(수십억 파라미터)에 대한 확장성 개선 필요

- **데이터 효율성**: 적은 검증 데이터로도 효과적인 하이퍼파라미터 최적화 가능 방법

- **하이퍼파라미터 선택 복잡성**: 초기 하이퍼파라미터 설정이 최종 성능에 미치는 영향에 대한 체계적 분석

**4. 새로운 응용 영역**

- **강화학습**: 양수준 최적화를 강화학습의 메타 학습에 적용

- **신경망 보정(Calibration)**: Bilevel 최적화를 통한 신경망 신뢰도 개선

- **도메인 일반화**: 메타 학습과 결합한 도메인 적응

**5. 최신 개선 방향 (2024-2025)**

- **함수형 프레임워크**: 함수 공간에서의 양수준 최적화로 강 convexity 가정 완화

- **Glocal 하이퍼기울도**: 전역적 신뢰성과 국지적 효율성을 동시에 달성

- **확률론적 안정성**: 확률론적 설정에서의 분산 감소 및 수렴성 보장 강화

***

### 결론

"Optimizing Millions of Hyperparameters by Implicit Differentiation"은 신경망 하이퍼파라미터 최적화의 **확장성 문제를 획기적으로 해결**한 논문입니다. 암시 함수 정리와 효율적인 역 헤시안 근사의 결합을 통해, 종전에 불가능했던 수백만 개의 하이퍼파라미터를 동시에 최적화할 수 있음을 입증했습니다.

**가장 중요한 기여**는 다음 세 가지입니다:

1. **상수 메모리 Neumann 근사**: 메모리 효율성 측면에서 기존 방법을 크게 개선

2. **이론과 실제의 연결**: IFT가 펼쳐진 미분화의 극한임을 보여줌으로써 개념적 통일성 제시

3. **일반화 성능 개선의 경로**: 검증 데이터의 적절한 할당을 통해 고차원 하이퍼파라미터 튜닝 시에도 일반화 성능 보장

최근 5년의 후속 연구는 이 논문의 아이디어를 다양한 방향으로 확장했으며, AutoML, 메타 학습, 신경망 보정 등 여러 실제 문제에 광범위하게 적용되고 있습니다. 특히 **분산 감소**, **비볼록 이론**, **단일 루프 알고리즘** 개발이 현재 활발한 연구 방향입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c56da1d8-ddbb-4b50-8f11-d96a65741ed9/1911.02590v1.pdf)
[2](https://arxiv.org/pdf/1911.02590.pdf)
[3](https://arxiv.org/pdf/2105.01637.pdf)
[4](http://arxiv.org/pdf/2303.17768.pdf)
[5](https://arxiv.org/pdf/2302.09726.pdf)
[6](http://proceedings.mlr.press/v139/ji21c/ji21c.pdf)
[7](https://arxiv.org/abs/2010.07962)
[8](https://academic.oup.com/nsr/article/11/8/nwad292/7440017)
[9](https://proceedings.neurips.cc/paper_files/paper/2024/hash/19ae2b95d3831c14373271112f189a22-Abstract-Conference.html)
[10](https://openreview.net/forum?id=mkmX2ICi5c)
[11](http://arxiv.org/pdf/2206.08558.pdf)
[12](https://arxiv.org/pdf/2109.14119.pdf)
[13](https://arxiv.org/pdf/1705.03071.pdf)
[14](http://arxiv.org/pdf/2201.04545.pdf)
[15](https://arxiv.org/abs/1903.01997)
[16](https://theory.stanford.edu/~valiant/papers/Implicit_Regularization_COLT.pdf)
