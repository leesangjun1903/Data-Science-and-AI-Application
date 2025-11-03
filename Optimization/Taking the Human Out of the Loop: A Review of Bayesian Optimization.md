# Taking the Human Out of the Loop: A Review of Bayesian Optimization

## 1. 논문의 핵심 주장과 주요 기여

본 논문은 **Bayesian Optimization(BO)**을 자동화된 설계 문제 해결의 강력한 도구로 제시합니다. 저자들의 핵심 주장은 다음과 같습니다.[1]

**핵심 주장:**
- 복잡한 고차원 설계 문제에서 인간의 개입을 제거하고 자동으로 최적 구성을 찾을 수 있다[1]
- 확률적 대리 모델(probabilistic surrogate model)과 획득 함수(acquisition function)의 결합이 효율적인 순차적 최적화를 가능하게 한다[1]
- 대리 모델의 선택이 획득 함수의 선택보다 최적화 성능에 훨씬 더 중요하다는 것을 보여준다[1]

**주요 기여:**
1. **통합적 체계화**: 매개변수 모델(parametric models), 비매개변수 모델(nonparametric models), 획득 함수를 체계적으로 정리[1]
2. **광범위한 응용 사례**: AB 테스팅, 추천 시스템, 로봇공학, 하이퍼파라미터 튜닝, 자동 머신러닝 등 다양한 분야의 실제 응용을 소개[1]
3. **이론과 실무의 교량**: 이론적 수렴성 증명과 실무적 고려사항을 균형있게 제시[1]

---

## 2. 해결 문제, 방법론 및 모델 구조

### 2.1 문제 정의

논문이 해결하고자 하는 근본적인 문제는:

$$
x^* = \arg \max_{x \in X} f(x)
$$

여기서 $$X$$는 설계 공간, $$f$$는 **블랙박스 함수**입니다. $$f$$는 다음의 특성을 갖습니다:[1]

- 닫힌 형태(closed form)가 없음
- 임의의 쿼리 지점 $$x$$에서만 평가 가능
- 평가는 노이즈가 있는 관측치 $$y \in \mathbb{R}$$를 생성: $$\mathbb{E}[y|x] = f(x)$$[1]
- 평가 비용이 매우 높음

**핵심 문제상황:**
- IBM ILOG CPLEX 솔버는 76개의 자유 매개변수를 가지며, 수동 조정은 불가능[1]
- 대규모 소프트웨어 라이브러리는 수백 개 이상의 상호작용하는 매개변수를 가짐[1]
- 하이퍼파라미터 튜닝 시 교차 검증 평가가 매우 비용이 높음[1]

### 2.2 Bayesian Optimization의 방법론

BO는 **모델 기반 순차적 최적화** 접근법입니다. 알고리즘의 핵심:[1]

**Algorithm 1: Bayesian Optimization**[1]

```
for n = 1, 2, ... do
  select new x_{n+1} by optimizing acquisition function α
    x_{n+1} = arg max_x α(x|D_n)
  query objective function to obtain y_{n+1}
  augment data D_{n+1} = {D_n, (x_{n+1}, y_{n+1})}
  update statistical model
end for
```

### 2.3 대리 모델(Surrogate Models)

#### **2.3.1 매개변수 모델**

**Beta-Bernoulli 모델:**

이항 반응에 대한 가장 단순한 모델입니다. 사전 분포:[1]

$$
p(w) = \prod_{a=1}^{K} \text{Beta}(w_a | \alpha, \beta)
$$

관측 데이터 $$D_n = \{(a_i, y_i)\}_{i=1}^n$$에서 사후 분포는:

$$
p(w|D) = \prod_{a=1}^{K} \text{Beta}(w_a | n_{a,1} + \alpha, n_{a,0} + \beta)
$$

여기서 $$n_{a,1}$$ = 약물 a의 성공 횟수, $$n_{a,0}$$ = 실패 횟수입니다.[1]

**Thompson Sampling (TS):**

$$
w \sim p(w|D_n), \quad a_{n+1} = \arg \max_a w_a
$$

TS는 사후 분포의 표본을 통해 자연스럽게 탐색-착취 균형을 달성합니다.[1]

**선형 모델:**

실값 응답에 대해:

$$
f(x_a) = x_a^T w, \quad y \sim \mathcal{N}(x_a^T w, \sigma^2)
$$

사전 분포는 공액 선행(conjugate prior)인 정규 역감마(Normal-Inverse-Gamma):[1]

$$
\text{NIG}(w, \sigma^2 | w_0, V_0, \alpha_0, \beta_0)
$$

사후 매개변수:

$$
w_n = \left(V_0^{-1} + X^T X\right)^{-1}(V_0^{-1}w_0 + X^T y)
$$

$$
V_n = \left(V_0^{-1} + X^T X\right)^{-1}
$$

**일반화 선형 모델(GLM):**

이항 데이터에 대해 로짓 링크 함수 사용:

$$
\mathbb{E}[y|x] = g^{-1}(x^T w) = \frac{1}{1 + \exp(-x^T w)}
$$

공액 사전이 없어 근사 추론(MCMC, Laplace 근사) 필요합니다.[1]

#### **2.3.2 비매개변수 모델 - Gaussian Process (GP)**

GP는 베이지안 선형 회귀를 커널 트릭으로 확장한 비매개변수 모델입니다.[1]

**GP 정의:**

$$
f \sim \text{GP}(m_0, k)
$$

유한 점의 집합에서:

$$
f|X \sim \mathcal{N}(m_0(X), K(X, X))
$$

$$
y|f \sim \mathcal{N}(f, \sigma^2 I)
$$

**사후 예측:**

테스트 점 $$x_*$$에서:

```math
\mu_n(x_*) = m_0(x_*) + k(x_*)^T(K + \sigma^2 I)^{-1}(y - m_0(X))
```

```math
\sigma_n^2(x_*) = k(x_*, x_*) - k(x_*)^T(K + \sigma^2 I)^{-1}k(x_*)
```

**핵심 커널:**

Matérn 커널 (부드러움 매개변수 $$\nu$$):

$$
k_{\text{Matérn}}(r) = \sigma_0^2 2^{1-\nu} \Gamma(\nu)^{-1} \left(\sqrt{2\nu}r\right)^\nu K_\nu\left(\sqrt{2\nu}r\right)
$$

제곱 지수 커널 ($$\nu \to \infty$$):

$$
k_{\text{sq-exp}}(r) = \sigma_0^2 \exp\left(-\frac{1}{2}r^2\right)
$$

**주변 가능도(Marginal Likelihood):**

$$
\log p(y|x_{1:n}) = -\frac{1}{2}(y-m_0)^T(K+\sigma^2I)^{-1}(y-m_0) - \frac{1}{2}\log|K+\sigma^2I| - \frac{n}{2}\log 2\pi
$$

이는 하이퍼파라미터 최적화에 사용됩니다.[1]

**계산 복잡도:** $$O(n^3)$$ (공분산 행렬 역함수), 실제 응용에서는 희소화 필요[1]

#### **2.3.3 희소 근사**

**Sparse Pseudoinput GP (SPGP):**

$$m \ll n$$ 유도 점으로 $$O(nm^2 + m^3)$$로 축소[1]

**Sparse Spectrum GP (SSGP):**

보흐너 정리를 이용한 스펙트럼 근사:

$$
k(r) = \frac{1}{(2\pi)^d} \int e^{iW^T x} s(W) dW \approx \frac{1}{m}\sum_{m=1}^M e^{iW_m^T x} e^{iW_m^T x'}
$$

**Random Forest 모델:**

결정 트리 앙상블로 $$O(n)$$에 가까운 복잡도, 범주형/조건부 입력 처리 용이하지만 외삽(extrapolation) 성능 취약[1]

---

## 3. 획득 함수(Acquisition Functions)

획득 함수 $$\alpha_n(x|D_n)$$는 다음 쿼리 포인트의 유틸리티를 정량화합니다.[1]

### 3.1 개선 기반 정책(Improvement-Based Policies)

**확률 개선(Probability of Improvement, PI):**

$$
\text{PI}(x|D_n) = \Phi\left(\frac{\mu_n(x) - \tau}{\sigma_n(x)}\right)
$$

여기서 $$\Phi$$는 표준 정규 누적 분포 함수, $$\tau$$는 목표값입니다. PI는 탐색이 부족해 과도한 착취 경향을 보입니다.[1]

**기대 개선(Expected Improvement, EI):**

$$
\text{EI}(x|D_n) = \mathbb{E}[\max(v - \tau, 0)|D_n]
$$

$$
= (\mu_n(x) - \tau)\Phi\left(\frac{\mu_n(x) - \tau}{\sigma_n(x)}\right) + \sigma_n(x)\phi\left(\frac{\mu_n(x) - \tau}{\sigma_n(x)}\right)
$$

여기서 $$\phi$$는 표준 정규 확률 밀도입니다.[1]

### 3.2 낙관주의 정책(Optimistic Policies)

**상한 신뢰도(Gaussian Process Upper Confidence Bound, GP-UCB):**

$$
\text{UCB}(x|D_n) = \mu_n(x) + \beta_n \sigma_n(x)
$$

여기서 $$\beta_n$$은 탐색 매개변수입니다. 탐색-착취 균형을 명시적으로 제어합니다.[1]

**엔트로피 검색(Entropy Search, ES):**

정보 이득을 최대화하는 보다 정교한 방법으로, 현재 불확실성을 기반으로 다음 평가를 선택합니다.[1]

### 3.3 Thompson Sampling의 일반화

$$
w \sim p(w|D_n), \quad x_{n+1} = \arg \max_x f_w(x)
$$

모든 획득 함수를 통합하는 원칙적 접근으로, 자연스러운 탐색-착취 균형을 제공합니다.[1]

***

## 4. 성능 향상(Performance Improvements)

### 4.1 대리 모델 선택의 중요성

논문의 **중요한 발견**: 대리 모델 선택이 획득 함수 선택보다 최적화 성능에 훨씬 더 중요합니다.[1]

**근거:**
- GP 모델은 불확실성을 원칙적으로 정량화
- Random Forest는 확장성이 좋지만 외삽이 취약
- 희소 근사는 계산 효율을 개선하지만 외삽 품질 저하

### 4.2 하이퍼파라미터 최적화

**Type II 최대우도(Empirical Bayes):**

$$
\theta^* = \arg \max_\theta \log p(y|X, \theta)
$$

준-뉴턴 방법(L-BFGS)으로 주변 가능도를 최적화합니다.[1]

**한계:**
- 관측 데이터 부족 시 과적합 위험
- 전체 베이지안 처리(hyperparameter marginalization)는 이론적 보장 부족[1]

### 4.3 병렬 Bayesian Optimization

**상수 거짓말쟁이(Constant Liar) 방법:**

$$
y_p = L \quad \forall p
$$

진행 중인 $$J$$ 평가에 대해 일정한 값을 귀속시킵니다.[1]

**판타지 샘플(Fantasy Sample) 방법:**

$$
\alpha(x|D_n, D_p) = \frac{1}{S}\sum_{s=1}^S \alpha(x|D_n, D_p^s)
$$

사후 예측 분포에서 $$S$$개의 결과를 샘플링하여 평행화합니다.[1]

### 4.4 고차원 문제 처리

**REMBO (Random Embedding Bayesian Optimization):**

효과적 차원성(effective dimensionality) $$d_e \ll D$$일 때:

$$
z^* \in \mathbb{R}^d: \quad f(x) = f(Az^*)
$$

무작위 행렬 $$A \in \mathbb{R}^{D \times d}$$로 저차원 부분공간에서 최적화합니다.[1]

***

## 5. 일반화 성능 향상(Generalization Performance Enhancement)

### 5.1 다중 작업 Bayesian Optimization(Multitask BO)

여러 관련 작업 $$\{f_1, \ldots, f_M\}$$을 동시에 최적화할 때, 한 작업의 정보가 다른 작업 성능 개선에 도움이 됩니다.[1]

**다중 출력 GP (Multioutput GP):**

입력-작업 쌍에 대한 공분산 함수:

$$
k(x, m; x', m') = \text{cov}(f_m(x), f_{m'}(x'))
$$

**내재 모델(Intrinsic Model of Coregionalization):**

$$
k(x, m; x', m') = k_x(x, x') k_m(m, m')
$$

한 작업에서 다른 작업으로 학습이 전이되어, 관측 효율성을 크게 향상시킵니다.[1]

### 5.2 비정상성(Nonstationarity) 처리

**비정상 커널 (Beta 변형):**

$$
x_d' = w_d(x_d) = \int_0^{x_d} \frac{t^{\alpha-1}(1-t)^{\beta-1}}{B(\alpha,\beta)} dt
$$

입력 공간을 변환하여 함수의 지역적 평탄도에 적응합니다.[1]

**이분산성(Heteroscedasticity) 처리:**

노이즈 프로세스가 입력에 따라 변할 때, 조각별 분할(partitioning)로 로컬 분산을 모델링합니다.[1]

### 5.3 조건부 공간(Conditional Spaces)

특정 변수가 활성/비활성인 구조화된 하이퍼파라미터 공간:

- **Random Forest**: 의사 결정 트리가 자연스럽게 비활성 변수 무시
- **Tree Parzen Estimator (TPE)**: 구조를 명시적으로 모델링
- **Gaussian Process**: 고정 길이 임베딩으로 확장 필요[1]

### 5.4 비용-민감 최적화

각 평가가 서로 다른 비용을 가질 때:

$$
\text{EI per unit cost} = \frac{\text{EI}(x|D_n)}{c(x)}
$$

시간적으로 효율적인 최적화를 달성합니다.[1]

***

## 6. 한계 및 도전 과제

### 6.1 높은 차원성(High Dimensionality)

**문제:**
- 정규 BO는 중간 차원($$D \lesssim 30$$)에 제한
- 전체 공간을 커버하려면 평가 횟수가 지수적으로 증가

**해결책:**
- REMBO: 저차원 부분공간 가정
- Random Search: 낮은 효과적 차원성 이용[1]

### 6.2 획득 함수 최적화

**문제:**
- 획득 함수 자체가 다중 봉우리(multimodal)
- 정확한 최적화를 보장하기 어려움
- 보조 최적화기(auxiliary optimizer)의 실패가 수렴성을 훼손[1]

**접근:**
- 분할 가능한 정사각형(DIRECT) 방법
- CMA-ES (Covariance Matrix Adaptation)
- 다중 시작 국소 검색[1]

### 6.3 하이퍼파라미터 추정의 불확실성

**문제:**
- 제한된 관측 데이터에서 하이퍼파라미터 과적합
- Type II MLE는 불확실성을 무시

**미해결:**
- 전체 베이지안 처리에 대한 이론적 수렴 보장 부족[1]

### 6.4 제약 조건 처리

**문제:**
- 제약 조건을 위반하는 구성이 실행 중 나타남

**방법:**
- 가중 기대 개선(wEI): $$\text{wEI}(x) = \text{EI}(x) \cdot \mathbb{P}(\text{constraint}|D_n)$$
- 적분 기대 조건부 개선(IECI)[1]

### 6.5 계산 비용

**GP 정확 추론:**
- 공분산 행렬 반전: $$O(n^3)$$
- 하이퍼파라미터 업데이트마다 재계산 필요
- 대규모 평가 예산에 부적합[1]

***

## 7. 논문의 영향과 향후 연구 고려사항

### 7.1 학문적 영향

1. **체계적 프레임워크 제공**: 매개변수/비매개변수 모델, 획득 함수를 통일된 관점에서 제시[1]

2. **이론-실무 연결**: 초기 일관성 증명부터 최근 유한 표본 경계까지 이론적 진전을 구조화[1]

3. **다양한 응용 통합**: AB 테스팅, 하이퍼파라미터 튜닝, 로봇공학 등 광범위한 적용 분야 제시[1]

### 7.2 향후 연구 시 고려할 점

**단기 과제:**

1. **고차원 확장성**: 1000+ 차원의 현실적 문제에 대한 방법론 개발 필요[1]

2. **하이퍼파라미터 불확실성**: 전체 베이지안 처리에 대한 이론적 보장 확보[1]

3. **병렬화 효율성**: 수백 개 병렬 평가에서의 최적 설계 점 선택[1]

**장기 방향:**

1. **딥러닝 통합**: 학습된 특성 맵(learned feature map)을 이용한 커널 설계로 고차원 문제 해결[1]

2. **적응형 모델링**: 데이터에 따라 자동으로 대리 모델 선택/전환하는 메타 알고리즘[1]

3. **전이 학습**: 다양한 작업/데이터셋 간 하이퍼파라미터 설정의 전이[1]

4. **실시간 응용**: 온라인 게임, 추천 시스템 등 스트리밍 데이터에서의 동적 최적화[1]

### 7.3 의료 영상 처리 맥락에서의 적용

귀사의 골 억제(bone suppression) 연구에 적용 가능한 통찰:

1. **모델 아키텍처 탐색**: BO를 이용한 U-Net 깊이, 채널 수, 정규화 강도의 동시 최적화로 일반화 성능 향상[1]

2. **데이터 효율성**: 의료 데이터 주석의 비용이 높으므로, BO의 표본 효율성이 핵심 장점[1]

3. **다중 데이터셋 전이**: Multitask BO로 여러 병원/기기의 X-ray 데이터셋 간 지식 공유[1]

4. **비용-성능 트레이드오프**: 모델 크기(추론 속도)와 정확도의 파레토 최적선 탐색[1]

---

## 요약

본 논문은 Bayesian Optimization의 이론, 방법론, 응용을 포괄적으로 다룬 리뷰 논문으로, **대리 모델의 신중한 선택과 획득 함수의 설계가 효율적인 자동화 최적화의 핵심**임을 강조합니다. 특히 데이터 효율성, 불확실성 정량화, 탐색-착취 균형이라는 세 축이 고차원 설계 문제의 해결을 가능하게 하며, 의료 영상 처리와 같은 전문 분야에서의 모델 일반화 성능 향상에 직접 적용 가능한 가치 있는 자산입니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a96d9f05-de5b-43e8-b5f7-814ccb4cad6f/Taking_the_Human_Out_of_the_Loop_A_Review_of_Bayesian_Optimization.pdf)
