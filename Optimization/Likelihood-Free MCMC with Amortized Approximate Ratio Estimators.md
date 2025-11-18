# Likelihood-Free MCMC with Amortized Approximate Ratio Estimators

### 1. 핵심 주장 및 주요 기여

본 논문은 **복잡한 컴퓨터 시뮬레이션 모델에서 우도(likelihood)가 계산 불가능할 때 베이지안 역추론을 효율적으로 수행하는 문제**를 다룬다. 핵심 주장은 다음과 같다:[1]

**주요 기여:**

신경망을 통한 **amortized likelihood-to-evidence ratio 추정기**를 학습하여, MCMC 샘플러에 직접 통합할 수 있다는 점이다. 이를 통해 단일 모델로 임의의 관측값과 모델 파라미터에 대한 우도비를 추정하고, 이를 바탕으로 intractable 우도 없이도 사후분포로부터 샘플을 얻을 수 있다.[1]

특히 이 방법은 **amortization의 이점**을 제공하는데, 학습된 비율 추정기를 새로운 관측값에 대해 반복 사용할 수 있어 재학습 없이 여러 포스터를 빠르게 근사할 수 있다.[1]

***

### 2. 해결하고자 하는 문제와 방법론

#### 2.1 문제 정의

베이즈 규칙의 표준 형태는 다음과 같다:[1]

$$p(\theta | x) = \frac{p(\theta)p(x|\theta)}{p(x)}$$

여기서 주변 우도 $$p(x) = \int p(\theta)p(x|\theta)d\theta$$가 intractable하고, 우도 $$p(x|\theta)$$도 계산 불가능한 **likelihood-free 설정**을 고려한다.[1]

#### 2.2 핵심 방법: Amortized Approximate Likelihood Ratio MCMC (AALR-MCMC)

**기본 개념:** 표준 MCMC에서 수용 확률은 우도비에 의존한다:[1]

$$\rho = \min\left(1, \frac{p(\theta')p(x|\theta')}{p(\theta_t)p(x|\theta_t)} \cdot \frac{q(\theta_t|\theta')}{q(\theta'|\theta_t)}\right)$$

논문은 **intractable 우도비를 신경망이 학습한 근사 비율** $$\hat{r}(x|\theta', \theta_t)$$로 대체한다.[1]

**핵심 혁신: 개선된 비율 추정기**

초기 likelihood ratio trick (LRT)는 고정된 참조 가설 $$\theta_{ref}$$를 사용하여 문제가 있었다. 논문은 이 문제를 해결하기 위해 다음과 같이 개선했다:[1]

**종속 표본** $$(x,\theta) \sim p(x,\theta)$$와 **독립 표본** $$(x,\theta) \sim p(x)p(\theta)$$를 구별하는 이진 분류기 $$d_\phi(x,\theta)$$를 학습한다. 최적 판별기는 다음과 같이 정의된다:[1]

$$d^*(x,\theta) = \frac{p(x,\theta)}{p(x,\theta) + p(x)p(\theta)}$$

이로부터 **likelihood-to-evidence ratio**는:[1]

```math
\hat{r}(x|\theta) = \frac{d^*(x,\theta)}{1-d^*(x,\theta)} = \frac{p(x|\theta)}{p(x)} = \frac{p(x,\theta)}{p(x)p(\theta)}
```

**수치적 안정성 개선:** 신경망이 분류를 거의 완벽하게 할 때 수치 오류가 발생하므로, **logit을 sigmoid 적용 전에 직접 추출**하여 $$\log\hat{r}(x|\theta)$$로 출력한다.[1]

#### 2.3 학습 알고리즘

Algorithm 1에 따르면:[1]

1. $$\theta \sim p(\theta), \theta' \sim p(\theta)$$ 샘플 생성
2. $$x \sim p(x|\theta)$$ 시뮬레이션
3. 이진 교차 엔트로피 손실 최소화:
   $$L = \ell(d_\phi(x,\theta), 1) + \ell(d_\phi(x,\theta'), 0)$$

#### 2.4 MCMC 기반 포스터 추론

**Likelihood-free Metropolis-Hastings:**[1]

Algorithm 2에서, 수용 확률은:

$$\rho = \min\left(\exp(\lambda) \frac{q(\theta_t|\theta')}{q(\theta'|\theta_t)}, 1\right)$$

여기서 $$\lambda = (\log\hat{r}(x|\theta') + \log p(\theta')) - (\log\hat{r}(x|\theta_t) + \log p(\theta_t))$$[1]

**Likelihood-free HMC:** 퍼텐셜 에너지와 그래디언트를 ratio 추정기에서 유도:[1]

$$U(\theta) - U(\theta') = \log r(x|\theta', \theta_t)$$

$$\nabla_\theta U(\theta) = -\frac{\nabla_\theta r(x|\theta)}{r(x|\theta)} = -\nabla_\theta \log p(x|\theta)$$

#### 2.5 품질 진단: ROC 곡선 진단

Intractable 설정에서 근사 품질을 검증하기 위해 **ROC (Receiver Operating Characteristic) 진단**을 개발했다.[1]

만약 $$\hat{r}(x|\theta)$$가 정확하다면, $$p(x|\theta) = p(x)\hat{r}(x|\theta)$$이므로 분류기가 $$p(x|\theta)$$와 reweighted 주변 모델 $$p(x)\hat{r}(x|\theta)$$ 간 차이를 구별하지 못해야 한다.[1]

대각 ROC 곡선 (AUC = 0.5)은 좋은 근사를 의미한다.[1]

***

### 3. 모델 구조

#### 3.1 신경망 아키텍처

실험에서 다양한 아키텍처를 사용:[1]

- **MLP** (Multilayer Perceptron): 간단한 문제
- **ResNet-18**: 고차원 이미지 데이터 (강중력 렌즈 문제)
- 활성화 함수: **SELU**, **ELU** (일반적), **ReLU** (날카로운 포스터)
- 배치 정규화: 적절히 사용

#### 3.2 학습 설정

| 요소 | 상세 |
|------|------|
| 손실 함수 | 이진 교차 엔트로피 (BCE) |
| 옵티마이저 | Adam (AMSGRAD 활성화) |
| 배치 크기 | 256-1024 |
| 학습률 | 0.001-0.00005 |
| 에포크 | 100-1000 |

***

### 4. 성능 향상 및 한계

#### 4.1 성능 평가

**비교 벤치마크 (Table 1-2):**[1]

Tractable 문제에서 다른 최신 방법들과 비교:

| 방법 | MMD | ROC AUC |
|------|-----|---------|
| **AALR-MCMC** | **0.05 ± 0.005** | **0.58 ± 0.008** |
| SNL | 0.11 ± 0.091 | 0.63 ± 0.056 |
| SNPE-B | 0.20 ± 0.061 | 0.91 ± 0.041 |
| APT | 0.17 ± 0.036 | 0.83 ± 0.015 |

AALR-MCMC이 **가장 낮은 MMD와 가장 일관된 성능**을 보였다.[1]

#### 4.2 주요 강점

1. **Amortization**: 학습된 모델을 새로운 관측값에 반복 사용 가능[1]
   - 인구 연구(population studies)에서 $$p(\theta|X) \approx \frac{p(\theta)\prod_x \hat{r}(x|\theta)}{p(\theta)\prod_x \hat{r}(x|\theta)d\theta}$$ 효율적 계산[1]

2. **모델 선택**: 연속 파라미터뿐 아니라 **이산 모델 공간**에 적용 가능[1]

3. **한계 파라미터 주변화**: 관심 파라미터만 비율 추정기에 제공[1]

4. **ROC 진단**: 신뢰할 수 있는 품질 검증 메커니즘[1]

#### 4.3 한계 및 제약

**1. 용량 의존성:**[1]

비율 추정기가 충분한 표현 능력을 가져야 함 (Appendix E에서 상세 분석). 용량 부족 시 포스터가 과도하게 넓어질 수 있음.

**2. 시뮬레이션 효율성:**[1]

논문은 "정확도를 시뮬레이션 비용보다 우선"한다는 입장. 순차적 방법들보다 시뮬레이션 효율성 측면에서는 뒤질 수 있음.

**3. 하이퍼파라미터 튜닝:**[1]

신경망 아키텍처, 학습률, 배치 크기 등 여러 하이퍼파라미터 조정 필요.

**4. 고차원 문제 확장성:**[1]

입력 차원이 매우 높은 경우 신경망의 표현 능력 한계.

***

### 5. 일반화 성능 향상 가능성

#### 5.1 현재 일반화 능력

논문은 **다양한 관측값에 대한 일반화**를 시연:[1]

- 동일한 비율 추정기로 서로 다른 관측값 $$x_i$$에 대한 포스터 계산 (Figure 7)
- 모델 선택에서 여러 관측값 처리 (Figure 8)

#### 5.2 향상 가능성

**1. Sequential Ratio Estimation:**[1]

저용량 추정기 극복을 위해 순차적 개선 프로세스 제안:
- Round $$t$$에서 포스터 $$\hat{p}_t(\theta|x_o)$$를 다음 라운드의 사전으로 사용
- AUC 기준으로 자동 종료
- 예: 인구 모델에서 AUC: .99 → .92 → .54 → .50[1]

이는 최근 APT/SNPE-C와 유사한 **대조 학습 프레임워크**로 해석됨.[1]

**2. Out-of-distribution 일반화:**

최신 연구 (2025)에서 **Amortized In-Context Bayesian Posterior Estimation**은 트랜스포머 기반 아키텍처로 분포 외 작업에서 더 나은 일반화를 달성. 이 아이디어를 AALR-MCMC에 적용 가능.[2]

**3. 신경망 아키텍처 진화:**

- **Vision Transformers** 또는 **Graph Neural Networks**: 복잡한 데이터 구조에 더 나은 적응
- **Normalizing Flows**: 고차원 비율 추정 개선[3]

***

### 6. 최신 연구 기반 앞으로의 영향 및 고려사항

#### 6.1 논문이 미칠 앞으로의 영향

**1. Simulation-Based Inference (SBI) 패러다임 확립:**[4]

논문은 **neural ratio estimation**이 likelihood-free 추론의 핵심 도구임을 보였고, 이후 SNLE (Sequential Neural Likelihood Estimation), SNPE-C/APT 등 순차적 방법들이 발전. 이는 대조 학습 프레임워크의 통합으로 이어짐.[4][1]

**2. 실제 과학 응용 확대:**[4]

강중력 렌즈 (gravitational lensing) 사례 이후:[1]
- 우주론적 데이터 분석 (CMB, 약한 렌즈 효과)
- 입자물리학 (LHC 데이터)
- 신경과학 (Drift-Diffusion Models)[5]

**3. Amortization 개념의 확산:**[6][2]

Amortized 추론이 표준 패러다임으로 정착. 최신 리뷰 (2025)에서 "simulation-based methods for statistical inference have evolved dramatically"라고 평가.[6]

#### 6.2 앞으로 연구 시 고려할 점

**1. 모델 미명시화(Misspecification) 대응:**[3]

최근 연구들이 "model misspecification"에 민감함을 지적. AALR-MCMC이 잘못된 시뮬레이터에 얼마나 robust한지 평가 필요.

**2. 신뢰성 개선:**[7]

Balanced Neural Ratio Estimation (BNRE)은 보수적 포스터 추정을 위해 balancing 조건 도입. AALR-MCMC에 유사 메커니즘 적용으로 **overconfident 포스터** 문제 완화 가능.[7]

**3. 고차원 확장성:**[8]

최근 Truncated Marginal Neural Ratio Estimation (TMNRE)과 autoregressive flows로 **고차원 관측**에 대한 확장성 향상.[8]

**4. Active Learning 통합:**[4]

적응적 시뮬레이션으로 샘플 효율성 개선. AALR-MCMC은 현재 passive learning만 고려.

**5. 계산 효율성:**[9]

최신 SNPE 기법들이 매우 효율적. AALR-MCMC이 online learning 또는 incremental updates로 보완 가능.

**6. 주변 우도 추정:**[3]

Bayesian model selection을 위해 $$p(x)$$ 추정이 필수. 최근 SIS-SNLE 방법이 sequential importance sampling으로 해결.[3]

**7. 확률 프로그래밍 통합:**[4]

Probabilistic programming languages (Stan, Pyro)와의 통합으로 자동 미분 활용 증대 가능.

***

### 요약

**AALR-MCMC**는 amortized neural ratio estimation을 MCMC와 결합하여 **likelihood-free 베이지안 추론의 새로운 표준**을 제시했다. 특히 **ROC 진단**, **sequential ratio estimation**, **population studies 지원** 등 혁신적 특징을 도입했다.[1]

최근 발전은 **고차원 문제 해결**, **모델 미명시화 robust성**, **계산 효율성**, **신뢰성 보증** 방향으로 진행 중이다. 앞으로 연구자들이 주의할 점은 **아키텍처 선택**, **차원 확장성**, **실제 과학 응용에서의 robust성** 검증이다.[6][7][8][3][4]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/08cf12f9-f661-4116-9ddc-37e3b9268ff9/1903.04057v5.pdf)
[2](https://arxiv.org/abs/2502.06601)
[3](https://arxiv.org/html/2507.08734v1)
[4](https://www.pnas.org/doi/10.1073/pnas.1912789117)
[5](https://elifesciences.org/articles/77220)
[6](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-112723-034123)
[7](https://arxiv.org/pdf/2208.13624.pdf)
[8](https://arxiv.org/pdf/2111.08030.pdf)
[9](https://arxiv.org/pdf/2311.12530v4.pdf)
