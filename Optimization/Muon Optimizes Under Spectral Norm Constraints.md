# Muon Optimizes Under Spectral Norm Constraints

---

## 1. Executive Summary (10문장 이내)

Muon optimizer [JJB+24]는 뛰어난 경험적 성능에도 불구하고 이론적 기반이 불분명했다.  
본 논문은 Muon을 **Lion- $\mathcal{K}$ optimizer 패밀리** [CLLL24]의 특수 케이스로 위치시킴으로써 이 공백을 메운다.  
구체적으로, Muon은 볼록 함수(convex function) $\mathcal{K}$를 **핵 노름(nuclear norm)** $\|\cdot\|\_{\text{tr}}$으로 설정한 Lion- $\mathcal{K}$에 해당함을 증명한다.  
분리된 가중치 감쇠(decoupled weight decay)를 갖춘 Muon은 가중치 행렬의 **스펙트럼 노름(spectral norm)에 제약**을 부과하는 최적화 문제를 암묵적으로 푼다.  
이는 Muon의 암묵적 정규화 효과를 이론적으로 설명한다.  
저자들은 결정론적 및 확률적 그래디언트 설정 모두에서 Muon의 수렴률을 공식적으로 확립한다.  
또한, Muon이 특정 스펙트럼 노름 제약 최적화 문제의 **KKT(Karush-Kuhn-Tucker) 점 집합으로 수렴**함을 보인다.  
이 이론적 프레임워크는 Lion- $\mathcal{K}$를 통해 다양한 볼록 함수 선택으로 일반화될 수 있다.  
실험적으로는 ResNet, ViT, LLaMA 등 다양한 아키텍처에서 이론적 예측이 검증된다.  
이 연구는 Muon을 이론적으로 근거 있는 딥러닝 optimizer로 정립하며, 향후 폭넓은 암묵적 정규화 알고리즘 설계에 방향을 제시한다.

---

## 1-1. 연구의 목적과 필요성

**목적:** Muon optimizer의 이론적 토대를 마련하고, 이를 더 넓은 최적화 프레임워크 안에서 통합적으로 이해하는 것.

**필요성 (p.1, Introduction):**
- Adam, AdamW 등 기존 적응형 optimizer는 이론적 기반이 잘 확립되어 있으나, Muon은 경험적 성능만 검증되었을 뿐 이론이 부재했음.
- Muon의 핵심 연산인 **Newton-Schulz 반복(Newton-Schulz iteration)**을 통한 직교화(orthogonalization)가 왜 효과적인지 설명할 이론이 없었음.
- 이론적 기반 없이는 Muon의 일반화(generalization), 수렴 보장, 및 체계적 확장이 불가능함.

> **💡 용어 설명**
> - **Newton-Schulz 반복**: 행렬의 제곱근 역행렬을 반복 계산으로 근사하는 수치 방법. Muon은 이를 사용해 그래디언트를 직교화함.
> - **직교화(Orthogonalization)**: 행렬의 모든 특이값(singular value)을 동일하게 만드는 변환. 정보의 등방성(isotropy)을 보장.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거/방법 | 관련 위치 |
|---|---|---|
| Muon = Lion- $\mathcal{K}$ (nuclear norm) | $\mathcal{K}(\mathbf{X}) = \|\mathbf{X}\|_{\text{tr}}$일 때 $\nabla\mathcal{K}(\mathbf{X}) = \text{msgn}(\mathbf{X})$ (Fact 2) | p.8-9, Section 6 |
| Muon은 스펙트럼 노름 제약 최적화를 암묵적으로 해결 | 분리된 가중치 감쇠 + 볼록 켤레(convex conjugate) 이론 → Eq. (5) | p.3, Section 6.1 |
| 결정론적 그래디언트에서 KKT 점수 수렴률 $O(1/\sqrt{T})$ | 이산 시간 Lyapunov 분석 + Proposition 4 | p.15, Theorem 3 |
| 확률적 그래디언트에서 수렴률 $O(1/\sqrt{T} + \sigma/\sqrt{n_{\text{batch}}})$ | 분산 유계 가정 + Lemma 5, 7 | p.16, Theorem 4 |
| Muon 반복이 KKT 점 집합으로 수렴 (a.s.) | LaSalle의 불변 원리(stochastic) | p.18-20, Theorems 5, 6 |
| Lion- $\mathcal{K}$ 프레임워크로 Muon의 자연스러운 일반화 가능 | 다양한 볼록 스펙트럼 함수 $\mathcal{K}$ 선택 (Table 1) | p.10, Section 6.2 |
| 실험적 제약 검증 | ResNet-18/50, ViT, LLaMA, Qwen에서 특이값 추적 | p.20-23, Figures 4-6 |

> **💡 용어 설명**
> - **KKT 조건(Karush-Kuhn-Tucker conditions)**: 제약 최적화 문제의 최적해에서 반드시 성립해야 하는 1차 필요 조건. 정류성, 원시 가능성, 이중 가능성, 상보성 조건으로 구성됨.
> - **Lyapunov 함수**: 동적 시스템이 균형점으로 수렴함을 증명하는 데 쓰이는 에너지 유사 함수. 단조 감소하면 수렴 보장.
> - **LaSalle의 불변 원리**: Lyapunov 함수가 단조 감소하는 경우, 시스템의 궤적이 특정 불변 집합으로 수렴함을 보이는 정리.

---

## 2-1. 해결 문제, 제안 방법, 모델 구조, 성능 향상 및 한계 상세 설명

### 📌 해결하고자 하는 문제

Muon optimizer [JJB+24]의 **이론적 수렴 보장 부재** 및 **암묵적 정규화 효과 미규명** (p.1).

---

### 📌 제안하는 방법 (수식 포함)

#### (A) Muon의 업데이트 규칙 (Eq. 2, p.3)

$$
\mathbf{M}_{t+1} = \beta_2 \mathbf{M}_t - (1 - \beta_2)\mathbf{G}_t
$$
$$
\widetilde{\mathbf{M}}_{t+1} = \beta_1 \mathbf{M}_t - (1 - \beta_1)\mathbf{G}_t
$$
$$
\mathbf{X}_{t+1} = \mathbf{X}_t + \eta_t\left(\text{msgn}(\widetilde{\mathbf{M}}_{t+1}) - \lambda\mathbf{X}_{t+1}\right)
$$

**기호 설명:**
- $\mathbf{X}_t \in \mathbb{R}^{n \times m}$: $t$ 번째 스텝의 가중치 행렬 (파라미터)
- $\mathbf{M}_t$: Polyak 모멘텀 (기울기의 지수 이동 평균)
- $\widetilde{\mathbf{M}}_t$: Nesterov 모멘텀 (추가 기울기 정보 반영)
- $\mathbf{G}_t$: $t$ 번째 스텝의 확률적 기울기 $\nabla\mathcal{F}(\mathbf{X}_t, \xi_t)$ 또는 결정론적 기울기 $\nabla\mathcal{F}(\mathbf{X}_t)$
- $\eta_t > 0$: 학습률 (learning rate)
- $\beta_1, \beta_2 \in [0,1)$: 모멘텀 계수 ($\beta_1 < \beta_2$)
- $\lambda \geq 0$: 가중치 감쇠 계수
- $\text{msgn}(\mathbf{X}) := (\mathbf{X}\mathbf{X}^\top)^{-\frac{1}{2}}\mathbf{X} = \mathbf{U}\text{sgn}(\boldsymbol{\Sigma})\mathbf{V}^\top$: 행렬 부호 함수 (Definition 2, p.8)

> **💡 용어 설명**
> - **Polyak 모멘텀**: 이전 기울기의 지수 가중 평균을 현재 업데이트에 더하는 기법. 진동 감소 효과.
> - **Nesterov 모멘텀**: 현재 위치에서의 기울기가 아닌, 모멘텀이 적용된 미래 위치에서의 기울기를 사용해 수렴을 가속하는 기법.
> - **행렬 부호 함수(msgn)**: 행렬의 특이값 분해(SVD) $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$에서 $\boldsymbol{\Sigma}$의 각 대각 원소에 signum 함수를 적용한 것. 핵 노름의 부분기울기(subgradient).

#### (B) Lion- $\mathcal{K}$의 일반 업데이트 규칙 (Eq. 3, p.3)

$$
\mathbf{M}_{t+1} = \beta_2 \mathbf{M}_t - (1 - \beta_2)\mathbf{G}_t
$$

$$
\widetilde{\mathbf{M}}_{t+1} = \beta_1 \mathbf{M}_t - (1 - \beta_1)\mathbf{G}_t
$$

$$
\mathbf{X}_{t+1} = \mathbf{X}_t + \eta_t\left(\nabla\mathcal{K}(\widetilde{\mathbf{M}}_{t+1}) - \lambda\mathbf{X}_{t+1}\right)
$$

**Muon = Lion- $\mathcal{K}$**: $\mathcal{K}(\mathbf{X}) = \|\mathbf{X}\|_{\text{tr}}$, $\nabla\mathcal{K}(\mathbf{X}) = \text{msgn}(\mathbf{X})$ (p.3, p.8-9)

#### (C) 암묵적 정규화 목적함수 (Eq. 4, p.3)

$$
\widehat{\mathcal{F}}(\mathbf{X}) := \mathcal{F}(\mathbf{X}) + \frac{1}{\lambda}\mathcal{K}^*(\lambda\mathbf{X})
$$

**기호 설명:**
- $\mathcal{K}^\*$: $\mathcal{K}$의 볼록 켤레(convex conjugate), $\mathcal{K}^\*(\mathbf{X}) := \sup_{\mathbf{Y} \in \mathbb{X}}(\langle \mathbf{X}, \mathbf{Y}\rangle - \mathcal{K}(\mathbf{Y}))$

> **💡 용어 설명**
> - **볼록 켤레(Convex Conjugate, Legendre-Fenchel Transform)**: 볼록 함수 $f$에 대해 $f^*(y) = \sup_x (\langle x,y\rangle - f(x))$로 정의되는 쌍대 함수. 핵 노름의 볼록 켤레는 스펙트럼 노름 단위 볼에 대한 지시 함수(indicator function).

#### (D) 암묵적 제약 최적화 문제 (Eq. 5, p.3) — **핵심 결과**

$$
\min_{\mathbf{X} \in \mathbb{X}} \mathcal{F}(\mathbf{X}) \quad \text{s.t.} \quad \|\mathbf{X}\|_{\text{op}} \leq \frac{1}{\lambda}
$$

**기호 설명:**
- $\|\mathbf{X}\|_{\text{op}} = \sigma_1(\mathbf{X})$: 스펙트럼 노름 (최대 특이값)
- $\frac{1}{\lambda}$: 제약 반경 (가중치 감쇠 계수의 역수)

> **💡 용어 설명**
> - **스펙트럼 노름(Spectral Norm)**: 행렬의 최대 특이값. 행렬이 벡터를 얼마나 늘릴 수 있는지의 최대 배율. 딥러닝에서 모델의 Lipschitz 상수와 관련됨.
> - **핵 노름(Nuclear Norm / Trace Norm)**: 행렬의 모든 특이값의 합. 행렬 랭크의 볼록 완화(convex relaxation)로 저랭크 정규화에 사용됨. 스펙트럼 노름의 쌍대 노름.

#### (E) KKT 점수 함수 (Eq. 6, p.3)

$$
\mathcal{S}(\mathbf{X}) := \|\nabla\mathcal{F}(\mathbf{X})\|_{\text{tr}} + \langle \lambda\mathbf{X}, \nabla\mathcal{F}(\mathbf{X})\rangle
$$

$\mathbf{X}^\star$이 KKT 점 $\Leftrightarrow$ $\|\lambda\mathbf{X}^\star\|_{\text{op}} \leq 1$ AND $\mathcal{S}(\mathbf{X}^\star) = 0$ (Proposition 2, p.11)

#### (F) 제약 집합 도달을 위한 Lyapunov 함수 (Eq. 7, p.4)

$$
\mathcal{V}_{\mathbb{B}}(\mathbf{X}) = \max\left(\|\mathbf{X}\|_{\text{op}} - \frac{1}{\lambda}, 0\right)
$$

선형 수렴률:

$$
\mathcal{V}_{\mathbb{B}}(\mathbf{X}_t) \leq \left(\prod_{s=0}^{t-1}(1 - \eta_s\lambda)\right)\mathcal{V}_{\mathbb{B}}(\mathbf{X}_0)
$$

#### (G) 내부 Lyapunov 함수

$$
\mathcal{V}_\mathcal{K}(\mathbf{X}, \mathbf{M}) = \mathcal{F}(\mathbf{X}) - \mathcal{F}^\star + \frac{c}{\lambda}\left(\|\mathbf{M}\|_{\text{tr}} - \langle\lambda\mathbf{X}, \mathbf{M}\rangle\right)
$$

**기호 설명:**
- $c$: 적절히 정의된 스칼라 상수
- $\mathcal{F}^\star$: 손실 함수의 하한값 (최솟값)
- $\langle \mathbf{X}, \mathbf{M}\rangle = \text{Tr}(\mathbf{X}^\top\mathbf{M})$: Frobenius 내적

---

### 📌 모델 구조

본 논문은 단일 신경망 아키텍처가 아닌 **optimizer 이론 프레임워크**를 제안. 주요 구성 요소:

```
Lion-K 프레임워크
├── K 선택 → Muon (K = nuclear norm)
│                → Lion (K = ℓ₁ norm)
│                → 일반화 variants (Table 1, p.10)
├── Nesterov 모멘텀 (β₁)
├── Polyak 모멘텀 (β₂)
├── 비선형 사전조건화 ∇K
└── 분리된 가중치 감쇠 λ
```

실험 아키텍처 (p.20-23): ResNet-18/50 (CIFAR-10/ImageNet), ViT-B/16, Qwen-100M, LLaMA-300M, LLaMA-0.5B

---

### 📌 성능 향상 및 한계

**성능 향상:**
- **수렴률 확립**: 결정론적 $O(1/\sqrt{T})$, 확률적 $O(1/\sqrt{T} + \sigma/\sqrt{n_{\text{batch}}})$ (Theorems 3, 4)
- **a.s. 수렴 보장**: 초기화와 무관하게 KKT 점 집합으로 거의 확실한 수렴 (Theorems 5, 6)
- **실험적 검증**: ResNet, ViT, LLaMA에서 약 400 스텝 내 제약 조건 달성 확인 (Figure 4)
- **Muon vs AdamW 비교**: LLaMA 0.5B에서 Muon이 스펙트럼 정규화된 특이값 분포를 보임 (Figure 6)

**한계 (p.24, Section 9):**
- 핵 노름의 비미분성(nondifferentiability)으로 연속 시간 분석을 직접 적용 불가 → 이산 시간 분석 별도 필요
- 일반 학습률 스케줄, 비매끄러운(nonsmooth) 목적함수로의 확장 미검증
- 단일 수렴점 보장 없음 (Remark 3)
- 대규모 실험 부족 (더 큰 모델, 다양한 태스크)

---

## 3. 각 주장에 페이지/Figure 번호 표시

| 주장 | 위치 |
|---|---|
| Muon = nuclear norm Lion- $\mathcal{K}$ | p.3, p.8-9, Section 6, Fact 2 |
| 암묵적 스펙트럼 노름 제약 (Eq. 5) | p.3, Section 6.1 |
| KKT 점수 함수 정의 (Eq. 6) | p.3, Section 7.1 |
| $\mathcal{V}_\mathbb{B}$ 선형 감소 | p.4, Section 7.4, Theorem 5 |
| $\mathcal{V}_\mathcal{K}$ 단조 감소 | p.4, Section 7.4, Theorem 6 |
| 결정론적 수렴률 $O(1/\sqrt{T})$ | p.15, Theorem 3 |
| 확률적 수렴률 | p.16, Theorem 4 |
| KKT 수렴 (a.s.) | p.18-20, Theorems 5, 6 |
| 토이 예제 제약 검증 | p.20-21, Figure 2, 3 |
| ResNet-18 제약 검증 | p.21-22, Figure 4 |
| 대규모 모델 제약 검증 | p.22, Figure 5 |
| LLaMA 특이값 분포 비교 | p.22, Figure 6 |
| K 일반화 실험 | p.22-23, Figure 7, Table 1 |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 🔵 저자가 직접 보고한 결과

**연구 주제:**
> "Muon이 nuclear norm을 사용하는 Lion- $\mathcal{K}$의 특수 케이스이며, 분리된 가중치 감쇠와 결합 시 스펙트럼 노름 제약 최적화를 암묵적으로 해결한다." (Abstract, p.1)

**방법 (저자 직접 기술):**

$$
\text{Muon} \equiv \text{Lion-}\mathcal{K} \text{ with } \mathcal{K}(\mathbf{X}) = \|\mathbf{X}\|_{\text{tr}}, \; \nabla\mathcal{K}(\mathbf{X}) = \text{msgn}(\mathbf{X})
$$

**결과 (저자 직접 기술):**
- 결정론적 설정: $\min_{1 \leq t \leq T} \mathcal{S}(\mathbf{X}_t) = O\left(\frac{1}{\sqrt{T}}\right)$ (Theorem 3, p.15)
- 확률적 설정: $\min\_{1 \leq t \leq T} \mathbb{E}[\mathcal{S}(\mathbf{X}\_t)] = O\left(\frac{1}{\sqrt{T}} + \frac{\sigma}{\sqrt{n_{\text{batch}}}}\right)$ (Theorem 4, p.16)
- $\mathcal{V}_\mathbb{B}(\mathbf{X}_t)$가 선형 속도로 감소 (p.4)
- ResNet-18에서 약 400 반복 내 $\|\mathbf{W}\|_{\text{op}} \leq \frac{1}{\lambda}$ 달성 (p.21, Figure 4)

---

### 🟠 나의 해석

1. **일반화 성능과의 연결:** 스펙트럼 노름 제약 ($\|\mathbf{X}\|_{\text{op}} \leq 1/\lambda$)은 네트워크의 Lipschitz 상수를 간접적으로 제한하며, 이는 일반화 오차 경계(generalization bound)와 연결될 가능성이 높음. 저자는 이를 명시적으로 주장하지 않았으므로 추론적 해석임.

2. **$\lambda$의 이중적 역할:** $\lambda$가 크면 더 강한 스펙트럼 정규화가 적용되나, 동시에 업데이트 크기를 축소시켜 학습을 느리게 할 수 있음. 이 트레이드오프에 대한 체계적 분석은 논문에서 부재함.

3. **AdamW 비교의 의미:** Figure 6에서 AdamW가 스펙트럼 정규화를 보이지 않는다는 점은, AdamW가 스펙트럼 노름 제약이 없는 $\ell_\infty$ 제약 최적화를 하기 때문임 [XL24]. 이는 두 optimizer의 귀납적 편향(inductive bias) 차이를 잘 보여줌.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

| 항목 | 문제점 | 위치 |
|---|---|---|
| ⚠️ ResNet-18 "약 400 스텝" | 구체적 수치가 단일 실험에서 관찰된 것으로, 통계적 반복 실험(multiple runs) 결과가 아님 | p.21, Figure 4 |
| ⚠️ LLaMA 0.5B 특이값 분포 비교 | 동일한 하이퍼파라미터 조건에서의 비교인지 명확히 기술되지 않음; 성능 비교(loss, perplexity 등) 부재 | p.22, Figure 6 |
| ⚠️ Figure 5의 다중 아키텍처 검증 | 에러바(confidence interval) 없이 단일 run 결과만 제시 | p.22, Figure 5 |
| ⚠️ $C_\mathcal{K} = \sqrt{\min(n,m)}$ 의존성 | 실제 대형 언어 모델의 행렬 크기에서 이 상수가 얼마나 문제가 되는지 분석 없음 | p.14, Lemma 3 |
| ❌ AdamW 대비 학습 성능 비교 없음 | 스펙트럼 제약 검증은 하나, downstream 태스크 정확도/perplexity 직접 비교 결과 없음 | p.20-23 전반 |
| ⚠️ Assumption 4 (반복별 감소 분산) | 실제 학습에서 충족 여부가 불명확한 강한 가정 | p.6, Assumption 4 |

> **💡 용어 설명**
> - **에러바(Error bar)**: 실험 결과의 불확실성 또는 분산을 시각화한 것. 없으면 재현성(reproducibility)을 신뢰하기 어려움.
> - **귀납적 편향(Inductive Bias)**: 알고리즘이 학습 데이터 외의 상황에서 특정 가정을 따르도록 하는 내재적 편향. optimizer마다 다름.

---

## 6. 논문이 답하지 않는 질문

1. **Q: 스펙트럼 노름 제약이 실제 테스트 정확도나 일반화에 직접적으로 얼마나 기여하는가?**
   - 제약 사실만 증명하고, 이것이 성능에 미치는 인과적 영향은 실험되지 않음.

2. **Q: 최적 $\lambda$ 값은 어떻게 선택해야 하는가?**
   - 이론적으로 $\lambda$는 제약 강도를 결정하나, 태스크별 최적 설정 방법론이 제시되지 않음.

3. **Q: Muon의 수렴이 단일 점으로 보장되는가?**
   - 저자 스스로 Remark 3 (p.20)에서 부정: KKT 점 집합으로 수렴할 뿐, 단일 점 수렴은 보장 안 됨.

4. **Q: 비매끄러운(nonsmooth) 손실 함수에서도 이론이 성립하는가?**
   - L-smoothness 가정(Assumption 2)에 의존하여 ReLU 등 비매끄러운 활성화 함수 상황에서의 이론적 보장 부재.

5. **Q: 임베딩 레이어, 어텐션 바이어스 등 비행렬 파라미터에 대한 Muon의 처리는?**
   - 실제 Muon 구현 [LSY+25]에서는 행렬 파라미터에만 적용하나, 이 부분 파라미터 처리의 이론적 분석 없음.

6. **Q: Newton-Schulz 근사 오차가 이론적 보장에 미치는 영향은?**
   - 실제 구현에서 $\text{msgn}$은 Newton-Schulz로 근사되나, 근사 오차가 수렴에 미치는 영향 분석 없음.

7. **Q: Muon이 다른 최신 optimizer(Sophia, SOAP 등)와 실제 대형 모델 학습에서 어떻게 비교되는가?**
   - 이론적 비교만 있고 직접적인 wall-clock time, 메모리 효율, 수렴 속도 실험 비교 없음.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.4): Muon 수렴 거동 개요

**내용:** 2차원 행렬 최적화 문제에서 Muon의 수렴 과정을 4가지 관점에서 시각화.
- **왼쪽 (특이값 궤적):** $(\sigma_1, \sigma_2)$ 공간에서 $\|\boldsymbol{\sigma}\|_\infty \leq 1/\lambda$ 제약 영역(초록)으로 빠르게 진입.
- **가운데 왼쪽 ( $\mathcal{F}(\mathbf{X})$ ):** 손실 함수는 비단조적(nonmonotonic) 진동 보임.
- **가운데 오른쪽 (Lyapunov):** $\mathcal{V}\_\mathbb{B}$는 외부에서, $\mathcal{V}_\mathcal{K}$는 내부에서 각각 단조 감소.
- **오른쪽 ( $\|\boldsymbol{\sigma}\|_\infty$ ):** 스펙트럼 노름이 $1/\lambda$ 이하로 유지됨.

**해석:** 손실이 비단조적이어도 Lyapunov 함수가 단조 감소하므로 수렴 보장. 이것이 이 논문의 핵심 통찰.

---

### Figure 3 (p.23): 제약 내/외 초기화에서의 수렴 검증

**내용:**
- **상단 패널 ($\lambda=1.25$, 제약 내부 초기화):** 두 궤적 모두 최적점으로 수렴. 손실에 비단조 스파이크 있으나 Lyapunov $\mathcal{H}$ 는 단조 감소.
- **하단 패널 ($\lambda=4$, 제약 외부 초기화):** 최적점이 제약 영역 밖에 있어 제약 경계(boundary)로 수렴. Lyapunov $\mathcal{V}_\mathbb{B}$ 단조 감소.

**해석:** 이론적 Theorem 5, 6의 실험적 검증. 초기화 위치에 무관한 수렴 보장을 시각적으로 확인.

---

### Figure 4 (p.23): ResNet-18 CIFAR-10에서 특이값 제약 달성 과정

**내용:** Muon ($\lambda=2.0$)으로 학습 중 ResNet-18 가중치 행렬들의 특이값 히스토그램 시계열.
- Iteration 0: 특이값들이 $\frac{1}{\lambda}=0.5$를 초과하는 경우 다수.
- Iteration 400: 거의 모든 특이값이 0.5 이하로 진입.
- Iteration 2000: 제약 조건이 안정적으로 유지됨.

**해석:** 이론 예측(선형 수렴으로 빠르게 제약 달성)이 실제 신경망 학습에서도 성립함을 보임. 약 400 스텝이라는 구체적 수치는 실용적 의미 있음.

---

### Figure 5 (p.24): 대규모 아키텍처에서 이론적 상한 검증

**내용:** ResNet-50 (ImageNet), ViT-B/16 (ImageNet), Qwen-100M, LLaMA-300M에서 최대 특이값 $\|\boldsymbol{\sigma}\|_\infty$의 학습 과정 추적. $\lambda=2.0$ (빨강)과 $\lambda=4.0$ (초록).

**해석:**
- 모든 모델/태스크에서 특이값이 $1/\lambda$ 이하로 수렴하는 이론 검증.
- $\lambda$가 클수록 더 강한 제약(더 낮은 상한) 적용됨을 확인.
- ⚠️ 단, 에러바 없는 단일 run으로 통계적 신뢰도 제한적.

---

### Figure 6 (p.24): LLaMA 0.5B에서 Muon vs AdamW 특이값 분포 비교

**내용:** LLaMA 0.5B 모델의 Query ($\mathbf{W}_Q$), Key ($\mathbf{W}_K$), Value ($\mathbf{W}_V$) 행렬에서 수렴 후 특이값 분포 히스토그램.
- **Muon:** 좁은 범위에 집중된 균등한 분포 (대부분 소형 특이값).
- **AdamW:** 넓게 퍼진 분포, 큰 특이값 존재.

**해석:**
- Muon의 스펙트럼 정규화가 실제 대형 언어 모델 학습에서 관찰됨.
- 균등한 특이값 분포는 행렬의 모든 방향에 균등한 정보 전달을 의미하며, 이것이 Muon의 성능 우위의 원인일 수 있음.
- ⚠️ 정성적 비교이며, 이 분포 차이가 성능(perplexity 등)에 미치는 영향의 정량적 분석 부재.

---

## 8. 결론 및 후속 연구

### 8-1. 저자 제시 결론 및 후속 연구

**저자 결론 (p.23-24, Section 9):**
- Muon은 nuclear norm Lion- $\mathcal{K}$의 인스턴스.
- 분리된 가중치 감쇠와 함께 스펙트럼 노름 제약 KKT 점으로 수렴.
- Lion- $\mathcal{K}$ 프레임워크를 통한 이론 기반 일반화 가능.

**저자 제시 후속 연구 방향 (p.24, Limitations):**
1. 더 넓은 볼록 함수 $\mathcal{K}$ 클래스 탐색.
2. 일반 학습률 스케줄, 비매끄러운 목적함수로 확장.
3. 더 큰 모델과 다양한 태스크에서 실험적 검증.

---

### 8-1. 모델의 일반화 성능 향상 가능성

본 논문이 직접 다루지 않았으나, 이론적 프레임워크에서 도출되는 일반화 관련 시사점:

**스펙트럼 정규화와 일반화 오차의 연결:**

Bartlett et al. (2017)에 따르면, $L$-층 신경망의 일반화 오차는

$$
\mathcal{O}\left(\frac{\prod_{l=1}^{L}\|\mathbf{W}_l\|_{\text{op}}}{\sqrt{n}} \cdot \text{복잡도 항}\right)
$$

에 비례한다. Muon이 $\|\mathbf{W}\_l\|_{\text{op}} \leq 1/\lambda$를 강제함으로써:
1. **각 층의 Lipschitz 상수** $\|\mathbf{W}\_l\|_{\text{op}}$를 제어 → 전체 네트워크의 Lipschitz 상수 제한.
2. **특이값 균등화**: Figure 6처럼 균등한 특이값 분포는 표현력을 전 방향에 고르게 분산시켜 과적합 방지 가능성.
3. **$\lambda$ 하이퍼파라미터**: 정규화 강도를 조절하는 역할로, 큰 $\lambda$는 강한 정규화 → 높은 편향(bias), 낮은 분산(variance) 트레이드오프.

**단, 이 연결은 추론적이며 본 논문에서 직접 증명하지 않음.** 일반화 경계를 엄밀히 도출하는 것이 중요한 후속 연구 주제.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 기여 | 본 논문과 관계 |
|---|---|---|---|
| **Lion** [CLH+23] | 2023 | 기호 탐색으로 발견된 efficient optimizer | Muon의 벡터 버전; Lion- $\mathcal{K}$ 이론의 출발점 |
| **Lion- $\mathcal{K}$** [CLLL24] | 2024 | Lion의 이론적 기반, constrained optimization 관점 | 본 논문이 직접 확장하는 프레임워크 |
| **AdamW implicit bias** [XL24] | 2024 | AdamW가 $\ell_\infty$ 제약 최적화를 암묵적으로 해결 | Muon의 spectral norm 제약과 유사한 접근; optimizer별 implicit bias 패턴화 |
| **Muon (원본)** [JJB+24] | 2024 | Newton-Schulz 기반 직교화 optimizer 제안 | 본 논문이 이론화하는 대상 |
| **Muon scalable** [LSY+25] | 2025 | Muon의 LLM 대규모 학습 적용 | 본 논문의 이론이 설명하는 실용적 구현 |
| **Old optimizer, new norm** [BN24] | 2024 | Adam, Shampoo, Muon을 steepest descent로 재해석 | 모멘텀 미포함 한계; 본 논문은 모멘텀 포함한 이론 제공 |
| **SOAP** [VMZ+25] | 2025 | Shampoo + Adam 결합 | Shampoo도 Schatten norm과 관련; 유사한 스펙트럼 정규화 관점 가능 |
| **MARS** [YLW+25] | 2025 | 분산 감소(variance reduction)를 활용한 LLM 학습 | Muon의 확률적 설정 개선에 참고 가능 |
| **Frank-Wolfe perspective** [SW25] | 2025 | Muon+가중치 감쇠를 Frank-Wolfe로 분석 | 동일 현상을 다른 렌즈로 봄; 본 논문의 Robbins-Monro 수렴과 상보적 |
| **[ALP+25, LH25, Kov25, SHH+25]** | 2025 | Muon 수렴 분석 (다양한 smoothness 가정) | 가중치 감쇠 미포함; 본 논문은 가중치 감쇠 포함 최초 이론 |

> **💡 용어 설명**
> - **Robbins-Monro 조건**: 확률적 근사 이론에서 $\sum \eta_t = \infty$, $\sum \eta_t^2 < \infty$를 만족하는 감소 학습률 조건. 확률적 수렴 보장을 위한 고전적 요건.
> - **Schatten norm**: 행렬 특이값에 적용되는 $\ell_p$ 노름. $p=1$이면 핵 노름, $p=\infty$이면 스펙트럼 노름.

**본 논문이 앞으로의 연구에 미치는 영향:**

1. **Optimizer 이론화의 표준 방법론 제시:** 경험적 optimizer를 Lion- $\mathcal{K}$ 프레임워크에 배치하고 Lyapunov 분석으로 수렴을 증명하는 방법론이 새로운 optimizer 분석의 표준이 될 수 있음.

2. **스펙트럼 정규화 설계 원칙:** 딥러닝 모델의 스펙트럼 제약이 implicit하게 달성 가능하다는 점에서, 명시적 스펙트럼 정규화(Spectral Norm Regularization, [Miyato et al. 2018]) 대신 optimizer 선택만으로 동일 효과 달성 가능성.

3. **Muon의 LLM 채택 이론 지원:** [LSY+25]에서 이미 대규모 채택이 시작된 Muon에 이론적 정당성 부여 → 산업계 채택 가속화 기대.

**앞으로 연구 시 고려할 점:**

1. **일반화 경계 도출:** Muon의 스펙트럼 제약이 PAC-Bayes bound나 Rademacher complexity에 미치는 영향을 정량화.

2. **$\lambda$ 자동 조정:** 고정 $\lambda$ 대신 태스크/레이어별 적응형 $\lambda$ 설정 방법론 연구.

3. **비행렬 파라미터 처리:** 임베딩, LayerNorm 등에 대한 이론 확장 또는 최적 보완 optimizer 조합 연구.

4. **Newton-Schulz 근사 오차 분석:** 실제 구현에서 $\text{msgn}$ 근사 오차가 수렴 보장에 미치는 영향의 엄밀한 정량화.

5. **비볼록 손실과 안장점(saddle point) 회피:** 현재 이론은 KKT 점 수렴만 보장하나, Muon이 안장점을 얼마나 효율적으로 회피하는지 분석 필요.

6. **분산 학습 환경 적용:** [LWC+24]의 분산 Lion 연구처럼, 분산 환경에서의 Muon 이론 확장.

---

## 참고 문헌

- **Lizhang Chen, Jonathan Li, Qiang Liu.** "Muon Optimizes Under Spectral Norm Constraints." arXiv:2506.15054v2 [cs.LG], 29 Sep 2025.
- **[CLLL24]** Lizhang Chen, Bo Liu, Kaizhao Liang, and Qiang Liu. "Lion secretly solves a constrained optimization: As Lyapunov predicts." ICLR 2024.
- **[JJB+24]** Keller Jordan, Yuchen Jin, Vlado Boza, et al. "Muon: An optimizer for hidden layers in neural networks." 2024.
- **[CLH+23]** Xiangning Chen, Chen Liang, Da Huang, et al. "Symbolic discovery of optimization algorithms." NeurIPS 2023.
- **[LSY+25]** Jingyuan Liu, Jianlin Su, et al. "Muon is scalable for LLM training." CoRR abs/2502.16982, 2025.
- **[XL24]** Shuo Xie and Zhiyuan Li. "Implicit bias of AdamW: $\ell_\infty$-norm constrained optimization." ICML 2024.
- **[BN24]** Jeremy Bernstein and Laker Newhouse. "Old optimizer, new norm: An anthology." CoRR abs/2409.20325, 2024.
- **[LH19]** Ilya Loshchilov and Frank Hutter. "Decoupled weight decay regularization." ICLR 2019.
- **[SW25]** Maria-Eleni Sfyraki and Jun-Kun Wang. "Lions and Muons: Optimization via stochastic Frank-Wolfe." CoRR abs/2506.04192, 2025.
- **[VMZ+25]** Nikhil Vyas, Depen Morwani, et al. "SOAP: improving and stabilizing Shampoo using Adam for language modeling." ICLR 2025.
- **[YLW+25]** Huizhuo Yuan, Yifeng Liu, et al. "MARS: unleashing the power of variance reduction for training large models." ICML 2025.
- **[Wat92]** G. A. Watson. "Characterization of the subdifferential of some matrix norms." *Linear Algebra and its Applications*, 170:33–45, 1992.
- **[Roc70]** R. Tyrrell Rockafellar. *Convex Analysis.* Princeton University Press, 1970.
