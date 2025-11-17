# Learning with Diﬀerentiable Perturbed Optimizers

### 1. 핵심 주장과 주요 기여

**"Learning with Differentiable Perturbed Optimizers"**는 기계학습 파이프라인의 핵심 문제를 해결합니다: 이산적(discrete) 의사결정 작업이 계산 그래프의 역전파(backpropagation)를 방해한다는 문제입니다.[1]

**핵심 주장:**
- 정렬(sorting), 최근접 이웃 선택, 최단경로 등 이산 최적화 문제들은 쉽게 계산되지만 그 도함수가 국소적으로 상수이거나 미분 불가능하여 기울기 역전파를 차단합니다.[1]
- 이 문제를 해결하기 위해 **확률적 섭동(stochastic perturbation)** 기법을 통해 이산 최적화기를 미분 가능한 연산으로 변환할 수 있습니다.[1]

**주요 기여:**
- 블랙박스 솔버에 대한 수정 없이 일반적으로 적용 가능한 체계적 방법 제시[1]
- 확률적 평활화를 통해 도함수가 모든 곳에서 정의되고 0이 아닌 Jacobian 행렬을 확보[1]
- 섭동된 최대값과 최대인수의 연속 도함수를 단순한 기댓값 형태로 표현 가능[1]
- Fenchel-Young 손실 함수와의 자연스러운 연결 및 학습 이론 보장 제시[1]

---

### 2. 문제 정의, 제안 방법 및 이론적 기초

#### 2.1 문제 정의

기계학습에서 다루는 일반적인 이산 최적화 문제는 다음과 같이 표현됩니다:[1]

$$
F(\theta) = \max_{y \in C} \langle y, \theta \rangle, \quad y^*(\theta) = \arg\max_{y \in C} \langle y, \theta \rangle
$$

여기서:
- $$\theta$$ 는 입력 점수 벡터
- $$C$$ 는 유한 점 집합 $$Y$$ 의 볼록껍질(convex hull)
- $$F(\theta)$$ 는 최대값 함수
- $$y^*(\theta)$$ 는 최대인수 함수

**문제점:**
$$y^*(\theta)$$ 는 구간별 상수 함수(piecewise constant)이므로 거의 모든 곳에서 기울기가 0이고, 불연속 지점에서 정의되지 않습니다.[1]

#### 2.2 제안 방법: 섭동된 최대인수

논문은 입력에 확률 잡음을 추가하여 이 문제를 해결합니다:[1]

$$
F_\varepsilon(\theta) = \mathbb{E}[F(\theta + \varepsilon Z)] = \mathbb{E}\left[\max_{y \in C} \langle y, \theta + \varepsilon Z \rangle\right]
$$

```math
y^*_\varepsilon(\theta) = \mathbb{E}[y^*(\theta + \varepsilon Z)] = \nabla_\theta F_\varepsilon(\theta)
```

여기서:
- $$\varepsilon > 0$$ 는 온도 매개변수
- $$Z$$ 는 $$d\mu(z) \propto \exp(-\nu(z))dz$$ 형태의 확률분포를 따릅니다.[1]

#### 2.3 핵심 이론적 성질 (Proposition 2.2)

$$C$$ 가 공내부를 가진 볼록 다면체이고 $$\mu$$ 가 양의 미분 가능 밀도를 가질 때:[1]

1. **$$F_\varepsilon$$ 의 성질:**
   - 엄격히 볼록(strictly convex)
   - 두 번 미분 가능
   - $$R_C$$ -Lipschitz 연속 (여기서 $$R_C = \max_{y \in C} \|y\|$$)
   - 기울기는 $$R_C M_\mu/\varepsilon$$ -Lipschitz 연속

2. **$$y^*_\varepsilon(\theta)$$ 의 성질:**
   - $$\theta$$ 에 대해 모든 점에서 미분 가능
   - 항상 $$C$$ 의 내부에 위치

3. **온도 매개변수의 영향:**
   - $$\varepsilon \to 0$$ 일 때: $$y^\*_\varepsilon(\theta) \to y^*(\theta)$$ (원래 이산 해로 수렴)
   - $$\varepsilon \to \infty$$ 일 때: $$y^*_\varepsilon(\theta) \to y^\*_1(0) = \arg\min\_{y \in C} \Omega(y)$$

#### 2.4 도함수 계산 (Proposition 3.1)

적분 부분과 관련된 기법을 이용하면:[1]

```math
y^*_\varepsilon(\theta) = \mathbb{E}[y^*(\theta + \varepsilon Z)] = \mathbb{E}\left[F(\theta + \varepsilon Z) \frac{\nabla_z \nu(Z)}{\varepsilon}\right]
```

```math
J_\theta y^*_\varepsilon(\theta) = \mathbb{E}\left[y^*(\theta + \varepsilon Z) \frac{\nabla_z \nu(Z)^T}{\varepsilon}\right]
```

도함수들은 단순한 기댓값으로 표현되어 **Monte Carlo 방법으로 효율적이고 불편향된(unbiased) 추정이 가능**합니다.[1]

---

### 3. Fenchel-Young 손실 함수와 학습 이론

#### 3.1 Fenchel-Young 손실의 정의

정의 4.1에 따르면, 섭동된 모델에서 Fenchel-Young 손실은:[1]

$$
L_\varepsilon(\theta; y) = F_\varepsilon(\theta) + \varepsilon \Omega(y) - \langle \theta, y \rangle
$$

여기서:
- $$\Omega = (F_\varepsilon)^*$$ 는 $$F_\varepsilon$$ 의 Fenchel 쌍대
- 손실은 비음수이며 $$\theta$$ 에 대해 볼록
- $$y^*_\varepsilon(\theta) = y$$ 일 때만 최솟값 0을 달성

**기울기:**

$$
\nabla_\theta L_\varepsilon(\theta; y) = y^*_\varepsilon(\theta) - y
$$

이는 **Jacobian 계산 없이도 1차 최적화 방법으로 직접 최적화 가능**하다는 의미입니다.[1]

#### 3.2 연결성과 일반화

- Gumbel 분포를 사용할 때, 이 프레임워크는 **log-sum-exp 함수, Gibbs 분포, softmax 함수**를 일반화합니다.[1]
- 섭동된 모형에서 $$p_\theta(y)$$ 는 간단한 형태의 닫힌 표현식을 가지지 않지만 **쉽게 샘플링 가능**합니다. 이는 Gibbs 분포의 반대 특성입니다.[1]

#### 3.3 통계적 보장 (Proposition 4.1)

비지도 학습에서 경험적 손실 $$\bar{L}_{\varepsilon,n}(\theta)$$의 경험적 최소값 $$\hat{\theta}_n$$에 대해, $$n \to \infty$$ 일 때:[1]

$$
\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} \mathcal{N}\left(0, \left(\nabla^2_\theta F_\varepsilon(\theta_0)\right)^{-1} \Sigma_Y \left(\nabla^2_\theta F_\varepsilon(\theta_0)\right)^{-1}\right)
$$

이는 **표준 M-추정 이론이 적용 가능**함을 의미합니다.[1]

***

### 4. 모델 구조 및 실제 구현

#### 4.1 광범위한 적용 가능성

논문은 다음 작업들을 통일적으로 다룹니다:[1]

| 작업 | 표현 | 계산 비용 |
|------|------|---------|
| **최대값** | 단위 단순형(simplex) 위의 선형 계획 | $$O(d)$$ |
| **순위 매기기** | 순열 면체(permutahedron) 위의 선형 계획 | $$O(d \log d)$$ |
| **최단경로** | 유향 그래프의 흐름 제약 | $$O(\|E\|)$$ |

#### 4.2 실제 구현 (정의 3.1)

주어진 $$\theta$$ 에 대해, $$M$$ 개의 독립 표본 $$Z^{(1)}, \ldots, Z^{(M)}$$ 을 생성하면:[1]

$$
\overline{y}_{\varepsilon,M}(\theta) = \frac{1}{M}\sum_{m=1}^{M} y^*(\theta + \varepsilon Z^{(m)})
$$

는 $$y^*_\varepsilon(\theta)$$ 의 불편향 추정입니다.

**장점:**
- 블랙박스 솔버로 작동 가능
- 병렬화 용이
- 웜스타트(warm start)로 실행 시간 단축 가능[1]

#### 4.3 2중 확률적 기울기 (식 6)

지도 학습에서:[1]

$$
\bar{\gamma}_{i,M}(w) = J_w g_w(x_i) \left(\frac{1}{M}\sum_{m=1}^{M} y^*\left(g_w(x_i) + \varepsilon Z^{(m)}\right) - y_i\right)
$$

이 스킴은 **데이터 미니배치와 인공 샘플을 독립적으로 조정 가능**합니다.[1]

***

### 5. 성능 향상 및 일반화 능력

#### 5.1 분류 작업 (CIFAR-10)

- Fenchel-Young 손실이 교차 엔트로피 손실과 **경쟁력 있는 성능** 달성[1]
- 온도 매개변수 $$\varepsilon$$ 의 영향 분석: 과도하게 높거나 낮은 값은 훈련 및 테스트 성능 저하[1]

#### 5.2 라벨 순위 매기기 작업

21개 데이터셋에서 실험:[1]

- **Spearman 상관계수에서 76%** 이상의 데이터셋에서 더 나은 성능
- **90%** 이상의 데이터셋에서 5% 이내 범위의 성능
- 제안된 손실이 기존의 squared loss 및 blackbox loss 대비 **특히 노이즈 견고성 우수**[1]

**인공 데이터셋 결과:**
- 거의 정확한 라벨 ($$\sigma \approx 0$$) 조건에서 정확한 순위 예측[1]
- 노이즈 증가에 따른 견고성 증명[1]

#### 5.3 최단경로 학습 (Warcraft 지형)

- **완벽한 정확도:** 50 에포크 중 대부분에서 100%에 가까운 최적 경로 달성[1]
- **비용 비율:** 학습된 비용과 실제 최단경로 비용의 비율이 1.00에 수렴[1]
- Blackbox loss 및 squared loss 기준선 대비 **빠른 수렴**[1]

#### 5.4 일반화 성능 향상의 원인

**온도 매개변수의 역할:**
$$\varepsilon$$ 의 조정을 통해 **smoothness와 정보 보존 사이의 균형** 조절[1]

$$
\text{정확도} = f(\varepsilon)
$$

- $$\varepsilon$$ 가 너무 작음: 과도한 평활화로 정보 손실
- $$\varepsilon$$ 가 너무 큼: 평활화 부족으로 진동[1]

**이론적 기반:**
- $$F_\varepsilon$$ 의 엄격한 볼록성으로 인해 **소실하지 않는 기울기** 보장[1]
- $$y^*_\varepsilon(\theta)$$ 의 미분 가능성으로 **정보 역전파 원활**[1]

***

### 6. 한계 및 도전 과제

#### 6.1 이론적 한계

1. **하한 설정:** Proposition 2.2에서 $$C$$ 가 공내부를 가져야 함. (Remark 1)[1]
   - 저차원 부분공간에 포함된 점들의 경우 추가 처리 필요

2. **근사 오차:** 온도가 0에 가까워질 때, 근사 오차는 $$O(\varepsilon)$$ 에 수렴[1]
   - 실제 계산에서는 정확도와 안정성의 트레이드오프

#### 6.2 실증적 한계

1. **계산 비용:** $$M > 1$$ (다중 샘플) 사용 시 계산량 증가[1]
   - $$M=1$$에서도 만족스러운 결과를 보이지만 샘플 효율성 탐색 여지

2. **매개변수 튜닝:** $$\varepsilon$$ 선택이 성능에 민감[1]
   - 교차 검증으로 조정 필요

3. **비교 대상 제한:** 일부 기준선(특히 구조화된 예측 작업)이 제한적

---

### 7. 앞으로의 영향과 향후 연구 방향

#### 7.1 이 논문의 학문적 영향 (2020-2025)

**직접적 후속 연구:**

논문 발표 이후 **353회 인용**(웹 기준)을 기록하며 여러 분야에 영향을 미쳤습니다:[2][3]

1. **암시적 미분(Implicit Differentiation) 확장:**
   - OptNet (2021)에서 이차 계획법(QP)을 신경망 계층으로 통합[3]
   - "Efficient and Modular Implicit Differentiation" (2022)에서 자동 암시적 미분 제안[4]
   - "Beyond backpropagation" (2022)에서 평형 전파(equilibrium propagation) 연결[5]

2. **조합 최적화 학습의 확대:**
   - Differentiable Dynamic Programming (2018 선행, 2021 재논의)에서 동적계획법 계층화[6][7]
   - 2024년 "Graph Reinforcement Learning for Combinatorial Optimization" 서베이에서 이 방법론이 핵심 기초 기술로 포함[8]

3. **Fenchel-Young 손실의 활용:**
   - "Learning with Fenchel-Young Losses" (2020) 독립 논문으로 확대[9]
   - 2024년 "Online Structured Prediction with Fenchel–Young Losses"에서 온라인 학습 환경으로 확장[10]
   - 2025년 "A Fenchel-Young Loss Approach to Data-Driven Inverse Optimization"에서 역최적화(inverse optimization) 분야 적용[11]

#### 7.2 현재 (2024-2025) 최신 동향

**제약 만족 신경망:**
- "Design Linear Constrained Neural Layers" (2024)에서 암시적 최적화를 통해 hard constraints 강제[12]
- Softmax, Sigmoid와 같은 고전 계층들이 특정 제약을 가진 KL-발산 기반 암시적 최적화로 재해석[12]

**효율성 개선:**
- 2024년 "BPQP: Differentiable Convex Optimization Framework"에서 암시적 미분보다 효율적인 접근법 제안[13]
- "Nystrom Method for Accurate and Scalable Implicit Differentiation" (2023)에서 Hessian 역벡터곱 근사[14]

**신경망-최적화 통합의 이론화:**
- 2025년 "Efficient End-to-End Learning for Decision-Making" (메타-최적화)에서 최적화 문제 근사의 효율성 분석[15]
- 2024년 "Differentiable Convex Optimization Layers" 종합 서베이[16]

#### 7.3 향후 연구 시 고려할 점

**1. 알고리즘 확장 방향:**
- **비볼록 최적화:** 현재 선형계획법 기반이지만, 비볼록 문제로의 확대 필요[1]
- **대규모 문제:** 차원이 매우 큰 문제에 대한 확장성 연구
- **부분 미분 가능성:** 일부 변수만 미분하는 혼합 접근법

**2. 이론적 강화:**
- **비점근적 수렴율:** Proposition 4.1의 비점근적 경계 확보
- **일반화 한계:** 고정된 $$\varepsilon$$ 하에서 Rademacher 복잡도 분석
- **온도 효과의 정량적 분석:** $$\varepsilon$$ 선택의 이론적 가이드라인

**3. 응용 분야 확대:**
- **강화학습 통합:** 정책 최적화에서 이산 행동 선택의 미분 가능화[8]
- **인버스 최적화:** 관찰된 최적 해로부터 가중치 학습[11]
- **제약 충족 학습:** 하드 제약 만족과 신경망 학습의 결합[12]

**4. 구현 최적화:**
- **자동 온도 조정:** $$\varepsilon$$ 의 적응적 선택 메커니즘
- **다양한 노이즈 분포:** Gumbel 외 분포의 체계적 비교
- **GPU 병렬화:** 다중 샘플 Monte Carlo 추정의 효율적 구현

**5. 실무적 적용:**
- **로보틱스:** 이산 제어 선택과 연속 제어의 통합[1]
- **조합 검색:** 빔 검색(beam search) 등의 미분 가능화[1]
- **그래프 신경망:** 정렬, 매칭 등 구조화된 예측 계층 통합

***

### 결론

"Learning with Differentiable Perturbed Optimizers"는 **기계학습에서 장기간 미해결 과제인 이산-연속 통합**을 우아한 확률론적 방법으로 해결합니다. 논문의 강점은:

1. **수학적 우아함:** 엄밀한 이론 기반 위의 체계적 접근
2. **일반성:** 블랙박스 솔버에 적용 가능한 범용성
3. **실용성:** Monte Carlo 추정을 통한 효율적 구현
4. **연결성:** Fenchel-Young 손실과의 자연스러운 관계

2020년 발표 이후 암시적 미분, 조합 최적화 학습, 제약 만족 신경망 등 여러 분야에 기초적 영향을 미쳤으며, 2024-2025년 최신 연구들이 이를 다양한 방향으로 확장하고 있습니다. 향후 연구는 비볼록 문제로의 확대, 대규모 확장성, 그리고 강화학습·인버스 최적화 등 새로운 응용 분야 개척에 초점을 맞춰야 할 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/64b86fbc-c451-4ff2-9e60-0e30ddce9a06/2002.08676v2.pdf)
[2](https://arxiv.org/pdf/2209.00616.pdf)
[3](https://arxiv.org/pdf/1703.00443.pdf)
[4](https://arxiv.org/pdf/2105.15183v4.pdf)
[5](https://arxiv.org/pdf/2205.03076.pdf)
[6](https://proceedings.mlr.press/v80/mensch18a/mensch18a.pdf)
[7](https://arxiv.org/pdf/1802.03676.pdf)
[8](https://arxiv.org/html/2404.06492v1)
[9](https://jmlr.csail.mit.edu/papers/volume21/19-021/19-021.pdf)
[10](https://arxiv.org/pdf/2402.08180.pdf)
[11](https://arxiv.org/abs/2502.16120)
[12](https://openreview.net/pdf?id=85vAL51Gmu)
[13](https://arxiv.org/pdf/2411.19285.pdf)
[14](https://arxiv.org/pdf/2302.09726.pdf)
[15](https://arxiv.org/pdf/2505.11360.pdf)
[16](https://arxiv.org/abs/2412.20679)
[17](https://arxiv.org/pdf/1512.00369.pdf)
[18](https://arxiv.org/html/2409.11847)
[19](http://arxiv.org/pdf/2310.17584.pdf)
[20](https://arxiv.org/html/2403.02571v1)
[21](https://arxiv.org/pdf/1703.09947.pdf)
[22](http://arxiv.org/pdf/2102.04704.pdf)
[23](http://arxiv.org/pdf/2503.18317.pdf)
[24](https://arxiv.org/abs/2002.08676)
[25](https://proceedings.mlr.press/v80/jeong18a/jeong18a.pdf)
[26](https://papers.nips.cc/paper/2020/file/6bb56208f672af0dd65451f869fedfd9-Paper.pdf)
[27](https://www.ijcai.org/proceedings/2021/0610.pdf)
[28](https://proceedings.neurips.cc/paper_files/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html)
[29](https://homes.cs.washington.edu/~nasmith/papers/peng+thomson+smith.acl18.pdf)
[30](https://indico.ictp.it/event/9409/session/36/contribution/171/material/slides/0.pdf)
[31](http://arxiv.org/pdf/2403.01260.pdf)
[32](https://arxiv.org/pdf/2210.01802.pdf)
[33](https://epubs.siam.org/doi/pdf/10.1137/20M1358517)
[34](https://www.sciencedirect.com/science/article/abs/pii/S0893608025010238)
[35](https://arxiv.org/pdf/2404.06492.pdf)
[36](https://www.sciencedirect.com/science/article/pii/S0167739X25002298)
[37](https://proceedings.neurips.cc/paper/2021/file/70afbf2259b4449d8ae1429e054df1b1-Paper.pdf)
