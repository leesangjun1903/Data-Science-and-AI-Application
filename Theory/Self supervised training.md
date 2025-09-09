# Self-training 가이드

다음 글은 Self-training을 중심으로 한 준지도 학습(SSL, Semi supervised Learning) 가이드입니다. 핵심 개념을 명확히 정리하고, 연구 수준의 예시 코드(PyTorch)까지 제공합니다.

### Semi-supervised learning
Semi-supervised learning은 일부는 라벨링된 데이터와 일부는 라벨링되지 않은 데이터를 함께 사용하여 AI 모델을 학습시키는 기법입니다.  
이는 지도학습과 비지도학습의 중간 형태로, 라벨이 부족해 완전한 지도학습이 어렵고, 순수 비지도학습만으로는 성능이 부족할 때 효과적입니다.

주요 특징은 다음과 같습니다:

- 소량의 라벨된 데이터를 사용해 모델의 기본 학습을 진행한 후, 대량의 라벨 없는 데이터에 대해 모델이 예측한 의사(가짜) 라벨을 생성해 추가 학습을 수행합니다.
- 이러한 반복적 학습으로 모델의 성능이 점진적으로 향상됩니다.

비용과 시간이 많이 소요되는 라벨링 작업을 줄여 실용적입니다.

대표적인 활용 분야는 이미지 분류, 음성 인식, 텍스트 분류 등 라벨 수집이 어려운 복잡한 작업입니다.

즉, 라벨 데이터를 적게 얻으면서도 대량의 비정형 데이터를 효과적으로 활용해 모델의 예측력을 높이고자 할 때 사용하는 학습 방법입니다.

## 무엇이 핵심인가
Self-training은 모델이 **unlabeled 데이터에 pseudo-label을 스스로 생성**해 다시 학습에 활용하는 프레임워크입니다. Entropy Minimization(EntMin), Pseudo-Label, Noisy Student 같은 기법들이 대표적입니다.[3][4][5][6][2]

## 용어 정리
- Pseudo-label: 교사 혹은 본인 예측으로 생성한 임시 레이블입니다.[5][2]
- Entropy Minimization: unlabeled 예측의 **엔트로피를 낮추어** 결정 경계를 저밀도 영역으로 밀어내는 정규화입니다.[7][3]
- Consistency Regularization: 약한/강한 증강 간 예측 일관성을 유지합니다.[4][2]

## 왜 동작하는가
SSL의 고전 가정은 세 가지입니다. 저밀도 분리, 클러스터 가정, 그리고 매니폴드 가정입니다. EntMin과 Pseudo-label은 저밀도 분리와 클러스터 가정을 활용해 경계를 데이터 밀도가 낮은 곳으로 이동시킵니다.[8][3][4]

## 대표 기법 한눈에
- Entropy Minimization (NIPS 2004): unlabeled의 예측 분포 $$p_\theta(y|x)$$ 엔트로피를 최소화합니다 [3][7].
- Pseudo-Label (ICML 2013): 고신뢰 예측을 hard label로 채택하여 지도 손실로 재학습합니다.[2][5]
- Noisy Student (CVPR 2020): 교사로 pseudo-label 생성, 학생은 **강한 노이즈(증강·dropout·stochastic depth)** 하에 학습, teacher-student를 반복합니다.[6][9]

## 수식으로 이해하기
- Entropy Minimization:

$$
  \mathcal{L}\_{\text{ent}} = \mathbb{E}_{x\in \mathcal{U}}\left[-\sum_{k} p_\theta(k|x)\log p_\theta(k|x)\right] \quad (\text{unlabeled 엔트로피 최소화})
  $$

[10][3]
- Pseudo-Label Cross-Entropy:

$$
  \hat{y}=\arg\max_k p_\theta(k|x),\;\; \mathcal{L}_{\text{pl}}=\mathbb{E}_{x\in \mathcal{U}} \mathbf{1}( \max_k p_\theta(k|x)\ge \tau)\cdot \text{CE}(p_\theta(\cdot|x),\hat{y})
  $$

- $( \hat{y} = \arg\max_k p_\theta(k|x) )$ : 입력 ( x )에 대한 모델 $( p_\theta )$의 확률 분포에서 가장 확률이 높은 클래스 ( k )를 예측된 의사 라벨(pseudo-label)로 선택한 것
- $( p_\theta(k|x) )$ : 파라미터 $( \theta )$를 가진 모델이 입력 ( x )에 대해 클래스 ( k )를 예측하는 확률
- $( \mathcal{U} )$ : 라벨이 없는(unlabeled) 데이터 샘플들의 집합
- $( \mathbf{1}(\max_k p_\theta(k|x) \ge \tau) )$ : 예측 확률의 최대값이 임계값 $( \tau )$ 이상일 때 1, 아니면 0인 지시 함수 (confidence threshold 역할)
- $( \text{CE}(p_\theta(\cdot|x), \hat{y}) )$ : 모델의 예측 분포와 의사 라벨 $( \hat{y} )$ 간의 크로스 엔트로피 손실
- $( \mathcal{L}_{\text{pl}} )$ : unlabeled 데이터에 대해 confidence threshold를 만족하는 샘플에만 의사 라벨을 부여하여 계산한 크로스 엔트로피 손실의 기댓값

즉, 의사 라벨링은 모델의 예측 확률이 특정 임계치 이상일 때 그 예측 클래스를 라벨로 삼아, 라벨이 없는 데이터에 대해 크로스 엔트로피 손실을 구하여 준지도 학습에 활용하는 방법입니다.

[4][5]
- Consistency (weak/strong aug):

$$
  \mathcal{L}_{\text{cons}}=\mathbb{E}_{x\in \mathcal{U}} \text{CE}\big(p_\theta(\cdot|a_w(x)),\;\text{stopgrad}(p_\theta(\cdot|a_s(x)))\big)
  $$

- $(\mathcal{L}_{\text{cons}})$: 일관성(consistency) 손실 함수
- $(\mathbb{E}_{x \in \mathcal{U}})$: 미라벨된 데이터 분포 $(\mathcal{U})$상에서의 기댓값(평균)
- (x): 미라벨된 입력 데이터 샘플
- $(a_w(x))$: 입력 (x)에 대해 약한 증강(weak augmentation)을 적용한 결과
- $(a_s(x))$: 입력 (x)에 대해 강한 증강(strong augmentation)을 적용한 결과
- $(p_\theta(\cdot|a(x)))$: 파라미터 $(\theta)$를 가진 모델이 증강된 입력 $(a(x))$에 대해 예측한 클래스 확률 분포
- $(\text{CE}(\cdot, \cdot))$: 크로스 엔트로피 손실 함수(cross-entropy loss)
- $(\text{stopgrad}(\cdot))$: 그래디언트가 역전파 되지 않도록 연결 차단(stop-gradient)하는 연산(weak 예측을 기준(target)으로 쓰고 강한 예측은 업데이트에 영향 받지 않음)

즉, 약하게 증강된 입력에 대한 모델의 예측 $(p_\theta(\cdot|a_w(x)))$를 정답처럼 보고, 거기에 대해 강하게 증강된 입력 예측 $(p_\theta(\cdot|a_s(x)))$이 최대한 유사해지도록 강제하는 손실입니다.

이 방법은 semi-supervised learning(반지도 학습)에서 unlabeled 데이터의 예측 안정성을 확보하고, 강한 증강 시 발생할 수 있는 왜곡에 모델이 적응할 수 있도록 돕습니다.

[2][4]

## 장단점과 함정
- 장점: 레이블 비용 없이 일반화를 높이고, 대규모 unlabeled를 쉽게 흡수합니다.[6][2]
- 단점: 잘못된 pseudo-label이 누적되는 confirmation bias, 도메인 쉬프트에서 라벨 노이즈 민감성이 있습니다.[10][2]
- 완화법: 임계값 $$\tau$$, 온도 조절, 불확실성 기반 필터링, 강한 증강과 학생에만 노이즈 주입, 커리큘럼식 샘플 선택이 효과적입니다.[4][6]

## 연구 동향 스냅샷
- Entropy/PL의 이론·강건화: f-divergence, Rényi, label-noise robust loss 연구가 이어집니다.[11][10]
- 최적 수송/할당 관점의 PL: OTAMatch가 pseudo-label을 최적 운송 과제로 풀어 threshold 한계를 완화합니다.[12]
- 의료영상/세그멘테이션: U-Net에 EntMin과 consistency를 결합해 라벨 희소 환경에서 성능 향상을 보였습니다.[13]

### f-divergence
f-divergence는 두 확률분포 (P)와 (Q) 간의 차이를 측정하는 함수로, 기본적으로 **볼록(convex) 함수 $(f: (0, \infty) \to \mathbb{R})$**를 사용해 정의됩니다. 이때 (f)는 (f(1) = 0)을 만족해야 하며, 분포 (P)가 (Q)에 대해 절대 연속일 때 다음과 같이 정의합니다:

#### 볼록함수 정의방식 :
- $(f(t) = t \log t)$ (Kullback-Leibler divergence의 generator 함수)
- $(f(t) = |t-1|)$ (Variation distance)
- $(f(t) = (t-1)^2)$ (Pearson $(\chi^2)$ divergence)
이 함수들은 모두 (t > 0)에서 볼록합니다. f는 보통 (f(1) = 0)을 만족하여야 divergence의 성질에 부합합니다.

따라서, f-divergence에서 f는 양의 실수범위에서 **볼록성(Convexity)**을 갖는 함수이며, 이 조건이 충족되어야 다양한 divergence 지표들을 생성할 수 있습니다.

```math
[D_f(P | Q) = \mathbb{E}_Q \left[ f\left(\frac{dP}{dQ}\right) \right]
]
```

여기서 $(\frac{dP}{dQ})$ 는 Radon-Nikodym 미분(즉, (P)와 (Q)의 밀도 비율)이고, 확률밀도 함수 (p(x), q(x))가 존재하면

```math
[D_f(P | Q) = \int q(x) f\left(\frac{p(x)}{q(x)}\right) d\mu(x) + f'(\infty) P[q=0]
]
```

의 형태로 적분 표현할 수 있습니다. 이때 $(f'(\infty) = \lim_{x \to 0} x f(1/x))$이며, $(P[q=0])$는 $(q(x) = 0)$인 영역에 대한 (P)의 질량입니다.

##### Radon–Nikodym theorem 
두 확률측도 μ와 ν가 있을 때, μ가 ν에 대해 절대연속일 경우 Radon-Nikodym 정리에 의해 μ는 ν에 대한 유일한 미분 함수 $(dμ/dν)$ 로 표현되며, 이 함수가 두 분포 사이의 상대적 밀도 역할을 하기 때문입니다.

Radon-Nikodym 미분은 두 확률분포가 같은 측도 공간에서 정의될 때, 물리적 의미로는 한 분포의 밀도가 다른 분포에 대해 어떻게 변화하는지를 나타내는 밀도 함수로 볼 수 있습니다.  
예를 들어, 수조의 부피가 기준 측도 μ이고, 그 안의 물의 무게 분포가 측도 ν라면, ν를 μ에 대해 표현할 때 Radon-Nikodym 미분이 물의 밀도에 해당한다는 비유적 설명이 있습니다.

##### 두 분포가 매우 가까울 경우(같을 경우) 왜 수렴하나?
피셔 정보 행렬(Fisher Information Matrix, FIM)은 파라미터 $(\theta)$ 에 대한 모델의 민감도를 나타내며, 확률분포 $( p(x; \theta) )$ 에 대해 다음과 같이 정의됩니다.

```math
[F(\theta) = \mathbb{E}{p(x;\theta)} \left[ \nabla\theta \log p(x;\theta) \nabla_\theta \log p(x;\theta)^T \right]
]
```

즉, log-likelihood 함수의 기울기(스코어 함수)의 공분산 행렬입니다.

두 분포 $( p(x;\theta) )$ 와 $( p(x;\theta + d\theta) )$ 가 매우 가까울 때, f-divergence는 파라미터 변화 $( d\theta )$ 에 대하여 다음 근사식으로 표현될 수 있습니다.

```math
[D_f(p(\theta) | p(\theta + d\theta)) \approx \frac{1}{2} d\theta^T F(\theta) d\theta
]
```

##### 근사방식 : Taylor 2nd Approximation
테일러 2차 근사는 함수 ( f(x) )를 점 ( a ) 근처에서 2차 다항식으로 근사하는 방법입니다. 구체적으로,

```math
[f(x) \approx f(a) + f'(a)(x-a) + \frac{1}{2} f''(a)(x-a)^2
]
```

로 표현하며, 1차 근사에 2차 미분항까지 포함하여 함수의 변화율과 곡률을 동시에 고려합니다.

다변수 함수라면,

```math
[f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T (\mathbf{x} - \mathbf{a}) + \frac{1}{2} (\mathbf{x} - \mathbf{a})^T H_f(\mathbf{a}) (\mathbf{x} - \mathbf{a})
]
```

여기서 $(\nabla f(\mathbf{a}))$는 그래디언트, $(H_f(\mathbf{a}))$는 헤시안 행렬(2차 미분 행렬)입니다.

이를 통해 원래 함수 곡면을 접하는 2차 곡면으로 근사하여, 최적화나 수치해석에서 매우 유용하게 활용됩니다.

테일러 정리에 따라,

```math
[D_f(p(\theta + d\theta)) = D_f(p(\theta)) + \nabla D_f(p(\theta))^T d\theta + \frac{1}{2} d\theta^T \nabla^2 D_f(p(\theta)) d\theta + o(|d\theta|^2)
]
```

여기서 첫 번째 항은 0 (동일한 분포에서 발산은 0), 두 번째 항(1차 도함수)도 0 (최소점이기 때문), 따라서 주요 기여는 2차 도함수에 의해 지배되고, 이 2차 도함수 행렬이 Fisher 정보 행렬 $(F(\theta))$ 임을 알 수 있습니다.

피셔 정보 행렬은 다음과 같이 정의됩니다:

```math
[F(\theta) = \mathbb{E}{p(\theta)}\left[ \nabla\theta \log p(X;\theta) \nabla_\theta \log p(X;\theta)^T \right]
]
```

또는 로그 우도 함수의 음의 기대 헤시안으로도 표현됩니다:

```math
[F(\theta) = - \mathbb{E}{p(\theta)}\left[ \nabla^2\theta \log p(X; \theta) \right]
]
```

이를 통해 $( D_f )$ 의 2차 미분에 해당하는 헤시안 행렬이 피셔 정보 행렬과 일치함을 알 수 있습니다.

즉, f-발산은 $(\theta)$ 주변에서 0차와 1차 항이 사라지고 2차 항만 남아, 2차 테일러 근사가 성립하는 것입니다.

즉, 모든 f-divergence는 미세한 차이에서는 **피셔 정보 행렬에 기반한 쌍대 거리(differential metric)**로 수렴합니다.

> 쌍대 거리(differential metric)는 원래 공간의 미분 구조를 쌍대공간에서 선형 함수들의 거리나 크기로 표현하는 계량이라 이해하면 됩니다. 이는 다양한 미분 연산자, 내적, 거리 개념을 엄밀하게 연결하는 도구입니다.

이것은 KL-다이버전스 등 특정 f-divergence에 대해 테일러 근사를 적용하면, 2차 미분항이 지배하기 때문에 자연스럽게 피셔 정보 행렬이 그 거리 척도 역할을 하기 때문입니다.

따라서, 두 확률분포가 매우 가까울 경우, f-divergence는 파라미터 공간에서 피셔 정보 행렬에 의해 정의되는 내적 형태의 거리 척도로 근사되어 표현됩니다.

요약하면,

- f-divergence 정의: $( D_f(p | q) = \int q(x) f(\frac{p(x)}{q(x)}) dx )$
- 피셔 정보 행렬: $( F(\theta) = \mathbb{E}{p(x;\theta)}[\nabla\theta \log p(x;\theta) \nabla_\theta \log p(x;\theta)^T] )$
- 미소 거리 근사: $( D_f(p(\theta) | p(\theta + d\theta)) \approx \frac{1}{2} d\theta^T F(\theta) d\theta )$

즉, 두 분포가 가까울수록 f-divergence는 피셔 정보 행렬에 의해 결정되는 거리 척도로 수렴합니다.

f-다이버전스는 통계 추정, 가설 검증, 그리고 최근에는 생성적 적대 신경망 (f-GAN) 학습 등 다양한 분야에서 핵심적인 역할을 합니다.

즉, f-divergence는 특정한 볼록 함수 (f)를 이용해 무수히 많은 종류의 분포 간 거리 혹은 차이를 일반화한 개념이고, KL-다이버전스, Hellinger distance 등이 모두 f-divergence의 특수한 경우입니다.  
또한, f-다이버전스들은 서로 다른 ( f )에 대해 다르지만, 두 분포가 매우 가까울 경우 모두 피셔 정보 행렬(Fisher information matrix)에 기반한 거리 척도로 수렴하는 특징을 가집니다.  
이는 작은 변화에서 모든 f-다이버전스가 피셔 거리의 상수배로 간주될 수 있음을 의미합니다.

### Rényi entropy
Rényi 엔트로피는 샤논 엔트로피를 일반화한 하나의 매개변수 (q)에 따른 엔트로피 척도로, 확률 분포의 불확실성(정보량)을 측정하는 다양한 방법을 통합합니다. (q) 값에 따라 샤논 엔트로피, 하틀리 엔트로피, 미니멈 엔트로피 등 여러 엔트로피 특수 경우를 포함합니다.

수학적으로, $(q \neq 1)$ 일 때 Rényi 엔트로피는 다음과 같이 정의되며:

```math
[S_q^{\mathrm{Rényi}} = \frac{1}{1-q} \log \left( \sum_i p_i^q \right)
]
```
확률 $(p_i)$들은 분포의 각 결과 확률입니다. $(q \to 1)$ 일 경우, 이 정의가 샤논 엔트로피로 연속적으로 수렴합니다.

#### 왜 수렴하는가?
L'Hôpital 법칙을 적용하면,

```math
[\lim_{q \to 1} S_q = -\sum_i p_i \ln p_i
]
```

로, 바로 샤넌 엔트로피의 정의와 일치하기 때문입니다. 즉, Rényi 엔트로피는 q에 대한 함수이고 q=1일 때 정의가 특이하지만, 극한값을 통해 샤넌 엔트로피로 자연스럽게 이어집니다.

이 엔트로피는 (q) 값 조정으로 희귀 이벤트나 빈번한 이벤트에 대한 민감도를 바꿀 수 있어, 복잡계 분석, 뇌 활동 평가 등에서 활용되며, 통계역학, 정보이론, 물리학 등 다양한 분야에서 응용됩니다.

요약하자면, Rényi 엔트로피는 확률 분포의 불확실성 평가에 유연성과 일반성을 제공하는 중요한 정보 이론적 도구입니다.

#### Shannon entropy
샤논 엔트로피(Shannon entropy)는 확률 분포에 따른 사건들의 불확실성, 즉 정보의 평균적인 불확실성을 수학적으로 측정하는 개념입니다.  
클로드 샤논이 1948년에 발표한 정보 이론의 핵심 개념으로, 확률 변수 X의 엔트로피 H(X)는

```math
[H(X) = - \sum_{i} p_i \log p_i
]
```

로 정의됩니다. 여기서 $(p_i)$ 는 각각의 사건이 일어날 확률입니다.

엔트로피 값이 클수록 정보의 불확실성이 크고, 이는 더 많은 정보를 포함함을 의미합니다.  
예를 들어, 공정한 동전 던지기(앞면과 뒷면 확률 1/2)에서는 엔트로피가 높고, 편향된 동전 던지기(예: 앞면 확률 9/10)에서는 엔트로피가 낮습니다. 이는 결과 예측이 어려울수록 더 많은 정보량이 필요함을 보여줍니다.

샤논 엔트로피는 데이터 압축, 암호학, 머신러닝 등 광범위한 분야에서 정보의 양과 불확실성을 정량적으로 이해하는 데 필수적입니다.

### Loss-robust against label noise
레이블 노이즈에 강인한(loss-robust against label noise) 손실 함수는, 노이즈가 있는 라벨 데이터에서도 결정 경계를 잘 유지하고 모델의 성능 저하를 막는 손실 함수입니다. 대표적 수식은 다음과 같습니다.

#### 대칭 손실 함수 (Symmetric Loss)
모든 클래스에 대해 손실 값을 더했을 때 일정한 상수 ( C )가 되는 함수입니다. 즉,

```math
[ \sum_{j=1}^{K} \ell(\mathbf{g}(\mathbf{x}), \mathbf{e}_j) = C
]
```

여기서 $(\mathbf{g}(\mathbf{x}))$ 는 예측, $(\mathbf{e}_j)$ 는 j번째 원-핫 라벨 벡터입니다.  
이 조건을 만족하면, 균일(label noise) 노이즈 확률 $(\eta < \frac{K-1}{K})$ 에서 노이즈에 대해 강인하다고 증명됩니다.

#### 노이즈 내성 조건 (Noise Tolerant Loss Condition)
노이즈가 있는 데이터와 없는 데이터에서 기대 위험(expected risk)을 최소화하는 예측기가 동일한 의사결정 경계를 만드는 것을 의미합니다. 즉, 노이즈가 학습에 미치는 영향을 제한합니다.

#### 예시 손실 함수

- Mean Absolute Error (MAE): 항상 일정한 값을 갖기 때문에 대칭 손실의 특징을 지니며 노이즈에 강인함.
- Cross Entropy (CE): 비대칭이고 무한대로 발산 가능하므로 노이즈에 취약함.

#### Loss-robust against label noise 설계 방법론

- 노이즈 전이 행렬(Noise Transition Matrix)을 이용해 노이즈 확률을 보정하는 방법
- 기존 손실에 레귤러리제이션을 추가해 노이즈 샘플 학습 정도를 조절하는 방법

즉, 라벨 노이즈에 강인한 손실 함수는 작은 노이즈율에서 손실 총합이 고정되는 대칭 손실이며, 보통 MAE 같은 함수가 이에 포함됩니다.  
이런 손실 함수를 사용하면 노이즈가 있어도 모델이 잘 학습됩니다. 상세 수식과 증명은 위 아카이브 논문에서 확인할 수 있습니다.

## 실전 레시피
- 데이터 구성: Labeled 소량, Unlabeled 대량입니다.[6][2]
- 교사-학생 설정: 교사는 깨끗한 조건으로 pseudo-label 생성, 학생은 강한 노이즈로 학습합니다.[9][6]
- 임계값과 스케줄: 초기에 높은 $$\tau$$로 보수적으로 시작하고 점차 낮춰 커버리지를 키웁니다.[2][4]
- 반복: 학생이 교사보다 좋아지면 교사로 교체하고 반복합니다.[9][6]

## PyTorch 예시 코드: EntMin + Pseudo-Label + Noisy Student
아래 코드는 CV/의료영상에 그대로 응용 가능한 연구 수준 템플릿입니다. 핵심은 교사로 pseudo-label 생성, 학생에 강증강·dropout, unlabeled에 대해 엔트로피/PL를 혼합하는 것입니다.[3][6]

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1) 모델 정의: 교사/학생 동일 구조, 학생만 노이즈 강하게
class ConvNet(nn.Module):
    def __init__(self, num_classes=10, p_drop=0.3):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p_drop),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p_drop),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        h = self.feat(x).flatten(1)
        return self.head(h)

def build_teacher_student(nc=10):
    teacher = ConvNet(nc, p_drop=0.0)   # 교사는 노이즈 없음
    student = ConvNet(nc, p_drop=0.3)   # 학생은 드롭아웃 등 노이즈
    return teacher, student

# 2) 증강: 약한/강한
import torchvision.transforms as T
weak_aug = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])
strong_aug = T.Compose([
    T.RandomResizedCrop(32, scale=(0.6, 1.0)),
    T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
    T.RandomHorizontalFlip(),
])

# 3) 손실 함수
ce = nn.CrossEntropyLoss(reduction='none')

def entropy_min_loss(logits):
    p = F.softmax(logits, dim=-1)
    ent = -(p * (p.clamp_min(1e-8).log())).sum(dim=-1)
    return ent.mean()

def pseudo_label_loss(student_logits, hard_targets, mask):
    loss_vec = ce(student_logits, hard_targets)
    return (loss_vec * mask).sum() / (mask.sum().clamp_min(1.0))

# 4) 교사로 pseudo-label 생성 (약한 증강, 온도/임계값 포함)
@torch.no_grad()
def generate_pseudo_labels(teacher, xu_w, tau=0.95, Ttemp=1.0):
    teacher.eval()
    logits = teacher(xu_w) / Ttemp
    probs = F.softmax(logits, dim=-1)
    conf, hard = probs.max(dim=-1)
    mask = (conf >= tau).float()
    return hard, mask, logits

# 5) 학습 루프 요약
def train_ssl(teacher, student, dl_labeled, dl_unlabeled, optimizer, epochs=100,
              w_pl=1.0, w_ent=0.1, tau_init=0.95, tau_final=0.7, Ttemp=1.0, device='cuda'):
    teacher.to(device).eval()
    student.to(device).train()

    for epoch in range(epochs):
        # 커리큘럼식 임계값 스케줄
        tau = tau_init + (tau_final - tau_init) * (epoch / max(epochs-1, 1))
        for (xl, yl), (xu, _) in zip(dl_labeled, dl_unlabeled):
            xl, yl = xl.to(device), yl.to(device)
            # 약/강 증강
            xu_w = weak_aug(xu).to(device)
            xu_s = strong_aug(xu).to(device)

            # 교사 pseudo-label
            hard, mask, _ = generate_pseudo_labels(teacher, xu_w, tau=tau, Ttemp=Ttemp)

            # 학생 forward
            logits_l = student(xl)
            logits_u = student(xu_s)

            # 지도 손실
            sup = ce(logits_l, yl).mean()

            # PL 손실
            pl = pseudo_label_loss(logits_u, hard.to(device), mask.to(device))

            # 엔트로피 최소화(마스크 밖 샘플도 포함, soft sharpening)
            ent = entropy_min_loss(logits_u)

            loss = sup + w_pl * pl + w_ent * ent
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        # 학생이 교사보다 좋아지면 교사 업데이트(EMA 또는 스냅샷)
        # 간단히 EMA 사용
        with torch.no_grad():
            m = 0.999
            for pt, ps in zip(teacher.parameters(), student.parameters()):
                pt.data.mul_(m).add_(ps.data, alpha=1-m)
```
- 포인트: 교사는 깨끗한 예측(약증강), 학생은 강한 노이즈로 일반화 유도, unlabeled에 PL과 EntMin 혼합을 적용합니다.[3][6]
- 실험 팁: 의료영상처럼 라벨 희소 환경에서는 EntMin이 수렴을 돕고 세그멘테이션에도 적용됩니다.[13][2]

## 고급 트릭과 안정화
- 온도와 임계값: 하드 라벨(교사 $$T\to0$$)과 학생 온도 상향이 안정적이라는 분석이 있습니다.[10][4]
- 불확실성 기반 필터링: Deep Ensemble/MC Dropout으로 신뢰도-불확실성을 결합해 필터링하면 노이즈에 강해집니다.[4]
- 최적수송 기반 할당: 배치 단위로 pseudo-label을 OT로 재할당하면 임계값 버림 손실을 줄일 수 있습니다.[12]

## Noisy Student 미니멀 파이프라인
- 단계: 1) 교사(라벨 데이터) 학습 → 2) 대규모 unlabeled pseudo-label 생성 → 3) 학생(더 큰 모델, 강증강·dropout·stochastic depth) 학습 → 4) 학생→교사 갱신 반복입니다.[9][6]
- ImageNet에서 정확도 및 강건성 향상 사례가 보고되었습니다.[6][9]

## 의료영상에 적용 시 체크리스트
- 클래스 불균형: 클래스별 동적 임계값(CPL)이나 리샘플링을 결합합니다.[2][4]
- 도메인 쉬프트: EntMin/PL 모두 쉬프트에 민감하므로 강건 손실 대안(Robust CE 등)을 고려합니다.[11][10]
- 세그멘테이션: 픽셀/패치 단위 pseudo-label, consistency와 EntMin을 같이 씁니다.[13][2]

## 더 읽을 거리
- Entropy Minimization 원전: SSL의 엔트로피 정규화 동기가 잘 설명되어 있습니다.[7][3]
- Pseudo-Label 원전: 간단하지만 강력한 베이스라인의 출발점입니다.[5][2]
- Noisy Student: 대규모 unlabeled와 노이즈 주입의 결합으로 성능·강건성 모두 개선합니다.[9][6]
- 포괄 서베이/리뷰: 큰 그림과 변형기법을 정리합니다.[4][2]
- 사용자가 공유한 한국어 요약 글: 흐름과 비교를 빠르게 잡는 데 유용합니다.[1]

## 마무리 한 줄
핵심은 “신뢰 가능한 pseudo-label을 만들고, 학생에 **노이즈와 일관성 제약**을 주며, **엔트로피를 낮추되 과신을 통제**”하는 것입니다. 적절한 임계값·온도·증강·반복 전략이 성패를 가릅니다.[3][10][6][4]

[1](https://beta.velog.io/@dust_potato/SSL-Paper-ReviewSelf-training-1.-Semi-supervised-Learningby-Entropy-Minimization-NIPS-2004)
[2](https://arxiv.org/pdf/2202.12040.pdf)
[3](http://papers.neurips.cc/paper/2740-semi-supervised-learning-by-entropy-minimization.pdf)
[4](https://arxiv.org/html/2408.07221v1)
[5](https://storage.googleapis.com/kaggle-forum-message-attachments/7371/pseudo_label_draft.pdf)
[6](https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.html)
[7](https://dl.acm.org/doi/10.5555/2976040.2976107)
[8](https://www.molgen.mpg.de/3659531/MITPress--SemiSupervised-Learning.pdf)
[9](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.pdf)
[10](https://openreview.net/pdf?id=1oEvY1a67c1)
[11](https://ieeexplore.ieee.org/document/10619617/)
[12](https://ieeexplore.ieee.org/document/10599208/)
[13](https://ieeexplore.ieee.org/document/9777614/)
[14](https://velog.io/@dust_potato/SSL-Paper-ReviewSelf-training-1.-Semi-supervised-Learningby-Entropy-Minimization-NIPS-2004)
[15](https://www.semanticscholar.org/paper/bb405d8a89b1e0e123d9765efb1a59912c202034)
[16](https://www.semanticscholar.org/paper/9db7c1cfbcb2a400887e8a616e6a42551216ebf2)
[17](https://www.mdpi.com/1099-4300/25/1/149)
[18](https://www.mdpi.com/2076-3417/14/12/4993)
[19](https://ojs.aaai.org/index.php/AAAI/article/view/20907)
[20](https://arxiv.org/abs/2405.05012)
[21](https://www.mdpi.com/2306-5354/11/9/865)
[22](http://arxiv.org/pdf/2405.00454.pdf)
[23](https://arxiv.org/pdf/2004.08514.pdf)
[24](http://arxiv.org/pdf/1804.05734.pdf)
[25](http://arxiv.org/pdf/2403.15567.pdf)
[26](https://arxiv.org/abs/1904.06487)
[27](https://www.mdpi.com/1099-4300/21/10/988/pdf)
[28](https://arxiv.org/html/2310.13022)
[29](http://arxiv.org/pdf/2404.12398.pdf)
[30](http://arxiv.org/pdf/2409.07292.pdf)
[31](http://arxiv.org/pdf/2503.13942.pdf)
[32](https://papers.nips.cc/paper/2740-semi-supervised-learning-by-entropy-minimization)
[33](https://www.scribd.com/document/733831832/Xie-Self-Training-With-Noisy-Student-Improves-ImageNet-Classification-CVPR-2020-paper)
[34](https://www.sciencedirect.com/science/article/abs/pii/S016786550800055X)
[35](https://pmc.ncbi.nlm.nih.gov/articles/PMC10449823/)
[36](https://icml.cc/virtual/2025/poster/44953)
[37](https://hoya012.github.io/blog/Self-training-with-Noisy-Student-improves-ImageNet-classification-Review/)
[38](https://www.sciencedirect.com/science/article/pii/S1877050921016082)
[39](https://velog.io/@9e0na/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-CV-Noisy-Student-Training-2020-Summary)
[40](https://proceedings.mlr.press/v206/he23b/he23b.pdf)
[41](https://deep-learning-study.tistory.com/554)

https://velog.io/@dust_potato/SSL-Paper-ReviewSelf-training-1.-Semi-supervised-Learningby-Entropy-Minimization-NIPS-2004

https://lv99.tistory.com/79
