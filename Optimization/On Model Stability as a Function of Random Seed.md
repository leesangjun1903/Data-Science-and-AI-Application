# On Model Stability as a Function of Random Seed

### 1. 핵심 주장과 주요 기여

이 논문의 **핵심 주장**은 동일한 모델 구조와 하이퍼파라미터를 가진 신경망이더라도 다른 랜덤 시드로 초기화되면 완전히 다른 해석 결과를 생성할 수 있다는 것입니다. 저자들은 이 문제가 모델의 일반화 성능뿐만 아니라 모델 해석 가능성에까지 중대한 영향을 미친다는 점을 보였습니다.[1]

**주요 기여**는 다음과 같습니다:[1]

첫째, 랜덤 시드가 모델 성능(Prediction Stability)과 해석 일관성(Interpretation Stability)에 미치는 영향을 정량적으로 측정했습니다. 저자들은 주의(Attention) 기반 해석, 그래디언트 기반 특성 중요도, LIME 기반 해석 등 세 가지 해석 방법을 활용하여 동일 모델의 서로 다른 인스턴스들이 40-60% 정도 다른 중요 해석 단위를 생성할 수 있음을 발견했습니다.[1]

둘째, **Aggressive Stochastic Weight Averaging (ASWA)**과 **Norm-filtered Aggressive SWA (NASWA)**라는 두 가지 최적화 기법을 제안했습니다. 이 방법들은 모델의 안정성을 평균 72% 향상시켰으며, 특히 Diabetes MIMIC 데이터셋에서는 89%의 상대적 개선을 달성했습니다.[1]

### 2. 해결하는 문제와 제안된 방법론

#### 2.1 핵심 문제점

논문이 해결하고자 한 문제는 다층 신경망의 **비결정적 특성**입니다. 신경망 학습 과정에서 발생하는 다양한 무작위성(랜덤 매개변수 초기화, 미니배치 샘플링, 드롭아웃 등)은 모델의 성능 뿐만 아니라 모델이 내린 결정에 대한 해석까지 불안정하게 만듭니다. 이는 모델 해석 가능성이 "모델 클래스"에 고유한 것인지 아니면 "특정 모델 인스턴스"에만 의존하는 것인지라는 근본적인 질문을 제기합니다.[1]

#### 2.2 측정 지표

저자들은 모델 안정성을 두 가지 차원에서 측정했습니다:[1]

**예측 안정성(Prediction Stability)** 지표:
- 평균(Mean)과 표준편차(Standard Deviation) 측정을 통해 여러 실행 간 정확도 변동 추적

**해석 안정성(Interpretation Stability)** 지표:
- **상대 엔트로피(Relative Entropy)**:

$$H = -\sum_{i}^{d} Pr_1(i) \log \frac{Pr_1(i)}{Pr_2(i)}$$

여기서 $Pr_1(i)$와 $Pr_2(i)$는 두 서로 다른 모델의 주의 분포입니다. 엔트로피가 높을수록 해석이 불일치합니다.[1]

- **자카드 거리(Jaccard Distance)**:

$$J = 1 - \frac{|A \cap B|}{|A \cup B|} \times 100$$

상위 n개의 가장 중요한 토큰 집합 간 교집합을 비교합니다.[1]

#### 2.3 제안된 방법: ASWA와 NASWA

**Aggressive Stochastic Weight Averaging (ASWA)**:[1]

기존 Stochastic Weight Averaging(SWA)을 확장한 방법으로, 매 배치 업데이트마다 가중치를 평균합니다:[1]

$$W_{\text{swa}} = W_{\text{swa}} + \frac{W - W_{\text{swa}}}{e \cdot n + i - 1}$$

여기서 $e$는 에포크 번호, $n$은 총 반복 횟수, $i$는 현재 반복 번호입니다. 저자들은 각 에포크 끝에 모델 가중치를 평균화된 가중치로 교체합니다. 이 접근법의 직관은 손실 표면의 가파른 영역(saddle point)을 피하면서 보수적으로 최소값에 도달하는 것입니다.[1]

**Algorithm 1: Aggressive Stochastic Weight Averaging**

```
Require: 
  e = Epoch number
  m = Total epochs
  i = Iteration number
  n = Total iterations per epoch
  α = Learning rate
  O = Stochastic Gradient optimizer

Function ASWA(e, m, i, n, α, O):
  e ← 0
  while e < m do
    i ← 1
    while i ≤ n do
      W_swa ← W_swa + (W - W_swa) / (e·n + i - 1)
      W ← W + α·∇O(W)
      i ← i + 1
    W ← W_swa
    e ← e + 1
```

**Norm-filtered Aggressive SWA (NASWA)**:[1]

가중치 변화의 노름 차이를 필터링하여 더욱 정교한 안정성을 달성합니다:[1]

$$\text{If } ||W - W_{\text{swa}}|| > \bar{N} : \quad W_{\text{swa}} \leftarrow W_{\text{swa}} + \frac{W - W_{\text{swa}}}{e \cdot n + i - 1}$$

$$\text{Else: } \quad N_s \leftarrow N_s \cup ||W - W_{\text{swa}}||$$

현재 반복의 노름 차이가 이전 반복들의 평균 노름 차이보다 클 때만 ASWA 가중치를 업데이트합니다. 이는 최적화 경로상의 큰 변화만을 평균화하여 더 안정적인 해를 찾는 데 기여합니다.[1]

**Algorithm 2: Norm-filtered Aggressive Stochastic Weight Averaging**

```
Require:
  e = Epoch number
  m = Total epochs
  i = Iteration number
  n = Total iterations per epoch
  α = Learning rate
  O = Stochastic Gradient optimizer
  N_s = List of previous norm differences

Function NASWA(e, m, i, n, α, O, N_s):
  e ← 0
  while e < m do
    i ← 1
    while i ≤ n do
      N_cur ← ||W - W_swa||
      N_mean ← mean(N_s)
      
      if N_cur > N_mean then
        W_swa ← W_swa + (W - W_swa) / (e·n + i - 1)
        N_s ← {N_cur}  // Reinitialize
      else
        N_s ← N_s ∪ {N_cur}
      
      W ← W + α·∇O(W)
      i ← i + 1
    W ← W_swa
    e ← e + 1
```

### 3. 모델 구조 및 실험 설정

#### 3.1 사용된 모델

논문은 두 가지 신경망 구조를 사용했습니다:[1]

- **CNN 기반 모델**: 300차원 FastText 임베딩, 32차원 필터, 커널 크기 1, 3, 5, 7
- **양방향 LSTM 모델**: 300차원 임베딩, 128차원 숨겨진 계층

두 모델 모두 다음의 주의 메커니즘을 포함했습니다:[1]
- **가법적 주의(Additive Attention)**:

$$\alpha_t = \frac{\exp(e_{t})}{\sum_s \exp(e_s)}, \quad e_t = v^T \tanh(W_1 h_t + W_2 s_{t-1})$$

여기서 $h_t$는 인코더 숨겨진 상태, $s_{t-1}$은 디코더 이전 상태입니다.[1]

- **스케일된 내적 주의(Scaled Dot-Product Attention)**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

여기서 $Q$, $K$, $V$는 쿼리, 키, 값입니다.[1]

#### 3.2 데이터셋

7개의 다양한 데이터셋에서 실험을 수행했습니다:[1]

| 데이터셋 | 작업 유형 | 평균 길이 | 훈련 크기 | 테스트 크기 |
|---------|---------|---------|---------|----------|
| IMDB | 이진 분류 | 179 | 12,500 | 12,500 |
| DiabetesMIMIC | 이진 분류 | 1,858 | 6,381 | 1,353 |
| SST | 이진 분류 | 19 | 3,034 | 3,321 |
| AnemiaMIMIC | 이진 분류 | 2,188 | 1,847 | 3,251 |
| AgNews | 이진 분류 | 36 | 30,000 | 30,000 |
| ADR Tweets | 이진 분류 | 20 | 14,446 | 1,939 |
| SNLI | 다중 분류 | 14 | 182,764 | 183,187 |

### 4. 성능 향상 및 실증적 결과

#### 4.1 예측 안정성 개선

**CNN 모델 결과**:[1]

표준 CNN 모델의 표준편차 평균이 1.5%였던 반면, ASWA와 NASWA 방법은 표준편차를 크게 감소시켰습니다. 특히 높은 변동성을 보이던 ADR Tweets 데이터셋에서 2.65%에서 0.54%로 **79% 감소**했습니다.[1]

| 데이터셋 | CNN | CNN+ASWA | CNN+NASWA |
|---------|-----|---------|----------|
| IMDB | $89.8 \pm 0.79$ | $90.2 \pm 0.25$ | $90.1 \pm 0.29$ |
| Diabetes | $87.4 \pm 2.26$ | $85.9 \pm 0.25$ | $85.9 \pm 0.38$ |
| SST | $82.0 \pm 1.01$ | $82.5 \pm 0.39$ | $82.5 \pm 0.39$ |
| Anemia | $90.6 \pm 0.98$ | $91.9 \pm 0.20$ | $91.9 \pm 0.19$ |
| AgNews | $95.5 \pm 0.23$ | $96.0 \pm 0.11$ | $96.0 \pm 0.07$ |
| ADR Tweets | $84.6 \pm 2.65$ | $84.4 \pm 0.54$ | $84.4 \pm 0.54$ |

**LSTM 모델 결과**:[1]

LSTM 기반 모델도 유사한 패턴을 보였으며, ASWA/NASWA 적용 시 표준편차가 현저히 감소했습니다:[1]

| 데이터셋 | LSTM | LSTM+ASWA | LSTM+NASWA |
|---------|------|-----------|------------|
| IMDB | $89.1 \pm 1.34$ | $90.2 \pm 0.32$ | $90.3 \pm 0.17$ |
| Diabetes | $87.7 \pm 1.44$ | $87.7 \pm 0.60$ | $87.8 \pm 0.55$ |
| SST | $81.9 \pm 1.11$ | $82.0 \pm 0.60$ | $82.1 \pm 0.57$ |
| Anemia | $91.6 \pm 0.49$ | $91.8 \pm 0.34$ | $91.9 \pm 0.36$ |
| AgNews | $95.5 \pm 0.32$ | $96.1 \pm 0.17$ | $96.1 \pm 0.10$ |
| ADR Tweets | $84.7 \pm 1.79$ | $83.8 \pm 0.45$ | $83.9 \pm 0.45$ |

#### 4.2 해석 안정성 개선

**주의 기반 해석의 엔트로피 감소**:[1]

DiabetesMIMIC 데이터셋의 경우, 표준 모델에서 매우 높은 엔트로피(불일치)를 보였지만, ASWA와 NASWA 적용 후 약 **60% 엔트로피 감소**를 달성했습니다. 이는 모델의 다양한 인스턴스들이 동일 입력에 대해 유사한 주의 분포를 생성하게 되었음을 의미합니다.[1]

엔트로피의 정의에 따라:

$$\text{Entropy 감소율} = \frac{H_{\text{baseline}} - H_{\text{ASWA/NASWA}}}{H_{\text{baseline}}} \times 100\%$$

DiabetesMIMIC의 경우, 기준 모델 대비 약 60% 감소를 달성했습니다.[1]

**그래디언트 기반 해석**:[1]

그래디언트 기반 특성 중요도를 다음과 같이 정의합니다:[1]

$$g_i = \frac{|\partial y / \partial x_i|}{\sum_j |\partial y / \partial x_j|}$$

여기서 $y$는 모델 출력, $x_i$는 입력 특성입니다. 그래디언트 기반 해석도 주의 기반 해석과 유사한 불안정성을 보였으며, NASWA 적용 후 Jaccard 거리가 유의미하게 감소했습니다.[1]

**LIME 기반 해석**:[1]

LIME(Locally Interpretable Model-agnostic Explanations) 기반 해석도 동일한 안정성 문제를 보였으며, NASWA 적용으로 상위 20개 중요 항목의 일관성이 향상되었습니다. LIME은 다음과 같이 정의됩니다:[1]

$$\text{explanation} = \arg\min_g L(f, g, \pi_x) + \Omega(g)$$

여기서 $f$는 원본 모델, $g$는 설명 가능한 모델, $\pi_x$는 $x$ 주변의 국소 커널입니다.[1]

### 5. 일반화 성능 향상 가능성

#### 5.1 안정성과 일반화의 관계

논문의 핵심 발견 중 하나는 **안정성 향상이 일반화 성능 감소를 초래하지 않는다**는 점입니다. ASWA와 NASWA는 모델의 성능을 유지하거나 심지어 개선하면서 표준편차를 감소시켰습니다.[1]

이를 수식으로 표현하면, 일반화 오차 바운드는:

$$\mathcal{L}_{\text{gen}} \leq \mathcal{L}_{\text{train}} + \mathcal{O}\left(\sqrt{\frac{\text{Var}(\theta)}{n}}\right)$$

여기서 $\text{Var}(\theta)$는 가중치의 분산, $n$은 샘플 수입니다. ASWA/NASWA는 $\text{Var}(\theta)$를 감소시켜 일반화 바운드를 개선합니다.[1]

#### 5.2 광범위한 신뢰도 구간에서의 안정성

저자들은 예측 신뢰도(confidence)를 0.1 단위의 구간으로 나누어 분석했습니다. 예측 신뢰도 구간별 표준편차 $\sigma_k$는 다음과 같이 정의됩니다:[1]

$$\sigma_k = \sqrt{\frac{1}{m} \sum_{j=1}^{m} (\text{acc}_{k,j} - \bar{\text{acc}}_k)^2}$$

여기서 $m$은 모델 인스턴스 수, $\text{acc}_{k,j}$는 신뢰도 구간 $k$에서 $j$번째 모델의 정확도입니다.[1]

표준 모델은 0.5 근처의 불확실한 영역에서 특히 높은 표준편차를 보였지만, ASWA/NASWA 모델은 모든 신뢰도 구간에서 더 낮고 안정적인 변동성을 보였습니다.[1]

#### 5.3 최적화 이론과의 연결

논문은 ASWA의 효과를 기하학적 관점에서 설명합니다. 비볼록 손실 표면에서 서로 다른 랜덤 시드는 다른 최적화 경로를 따르며, 일부는 안장점(saddle point) 근처를 통과합니다.[1]

Hessian 행렬 $H$를 기반으로 안정성을 분석하면:

$$\text{Stability} \propto \frac{1}{\lambda_{\max}(H)}$$

여기서 $\lambda_{\max}(H)$는 Hessian의 최대 고유값입니다. ASWA는 가중치 평균화를 통해 이러한 불안정한 영역을 피하고, 보다 안정적인 최소값에 도달하도록 유도합니다.[1]

이러한 메커니즘은 wider optima(더 넓은 최소값)으로의 수렴과 관련이 있으며, 이는 일반화 성능 향상과 관련된 Izmailov et al. (2018)의 이론과 일치합니다:[1]

$$\mathcal{L}_{\text{SWA}} \leq \mathcal{L}_{\text{SGD}} + \mathcal{O}(\sqrt{\text{width}})$$

### 6. 논문의 한계

#### 6.1 모델 특이성

실험이 CNN과 LSTM에 주로 집중되어 있으며, 이후 Transformer 기반 대규모 모델(현재 주류인 BERT, GPT 등)에 대한 검증이 필요합니다.[2]

#### 6.2 계층별 분석의 부재

논문은 모델 전체의 안정성을 다루었지만, 서로 다른 계층별로 불안정성이 어떻게 분포하는지에 대한 상세 분석이 부족합니다.[1]

#### 6.3 1차 신호에 제한

저자들이 직접 언급했듯이, ASWA와 NASWA는 1차 기반 신호(그래디언트)에 의존합니다. Hessian 같은 2차 정보를 활용하면 더 나은 안정성을 달성할 수 있을 것으로 예상됩니다.[1]

#### 6.4 계산 비용

ASWA/NASWA는 추가적인 가중치 평균화 연산이 필요하지만, 이로 인한 학습 시간 증가에 대한 구체적인 분석이 부족합니다.[1]

### 7. 앞으로의 연구에 미치는 영향

#### 7.1 모델 해석 가능성 연구에 미친 영향

이 논문은 **모델 해석 가능성 분야에서 중요한 신호**를 보냈습니다. 최근 연구들은 주의 메커니즘이나 다른 해석 방법의 신뢰성 문제를 더 심각하게 고려하기 시작했습니다. 특히 2024-2025년의 연구들은 해석 방법의 노이즈에 대한 견고성(robustness)을 측정하는 데 주목하고 있습니다.[3][4][5]

#### 7.2 대규모 언어 모델(LLM)에서의 랜덤 시드 문제

최근 2025년 연구에서는 **대규모 언어 모델 미세조정에서 랜덤 시드의 영향**을 본격적으로 조사했습니다. 이 연구는 GLUE와 SuperGLUE 벤치마크에서 매크로 및 미크로 수준의 성능 변동성을 체계적으로 평가했으며, 모델 예측의 일관성(consistency)을 측정하는 새로운 지표를 도입했습니다:[2]

$$\text{Consistency} = \frac{1}{m(m-1)} \sum_{i \neq j} \text{Acc}(y_i, y_j)$$

여기서 $y_i$, $y_j$는 서로 다른 시드로 훈련된 모델의 예측입니다.[2]

#### 7.3 안정성과 초기화의 중요성

2024년 논문 "On the Impacts of the Random Initialization in the Neural Network"은 **초기화 전략의 일반화 성능에 미치는 영향**을 신경 접선 커널(Neural Tangent Kernel, NTK) 이론 내에서 분석했습니다. 이는 본 논문의 주장을 이론적으로 뒷받침합니다.[6]

NTK 이론에서 일반화 오차는:

$$\mathcal{L}_{\text{gen}}(\theta) \sim \mathcal{O}(\sqrt{\text{Var}(\theta_{\text{init}})} / n)$$

이는 초기화의 분산이 일반화 성능에 직접 영향을 미침을 보여줍니다.[6]

#### 7.4 제한된 레이블 데이터에서의 안정성

2024년 설문 논문은 **적은 레이블 데이터를 사용하는 학습(프롬프팅, 컨텍스트 내 학습, 메타러닝, 퓨샷 학습 등)에서 랜덤 효과에 대한 민감성**을 강조했습니다. 이들 방법들은 학습 과정의 제어되지 않은 무작위성으로 인해 매우 불안정하며, 결과 변동성이 크다고 보고했습니다:[7]

$$\text{Variability}_{\text{few-shot}} = \sigma(\text{Acc}_{1}, \text{Acc}_{2}, \ldots, \text{Acc}_{m})$$

여기서 $m$은 서로 다른 랜덤 시드로 실행한 횟수입니다.[7]

#### 7.5 가중치 평균화의 일반화 이론

2024년 ICML 논문에서는 **Stochastic Weight Averaging의 일반화 이론**을 비볼록 설정과 대체 샘플링(sampling without replacement) 조건에서 엄밀하게 분석했습니다. 일반화 바운드는:[8]

$$\mathbb{E}_S[\mathcal{L}_{\text{gen}}] \leq \mathcal{L}_{\text{train}} + \frac{C \cdot \text{Var}(W_{\text{swa}})}{\sqrt{n}}$$

이는 본 논문의 ASWA 접근법의 효과성을 이론적으로 검증하는 방향으로 발전했습니다.[8]

#### 7.6 초기화 전략의 견고성

2024년 연구 "If You Want to Be Robust, Be Wary of Initialization"은 **가중치 초기화 전략이 적대적 공격에 대한 견고성(robustness)에 영향을 미친다**는 새로운 통찰을 제공했습니다. 초기화 전략에 따라 적대적 예제에 대한 저항성에 최대 50%의 차이가 발생할 수 있음을 보였습니다:[9]

$$\text{Robustness}_{\text{Adv}} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}[\text{Pred}(x + \delta) = y]$$

여기서 $\delta$는 적대적 섭동입니다.[9]

### 8. 향후 연구 시 고려할 점

#### 8.1 현대 신경망 아키텍처 확장

**Transformer 및 Vision Transformers**: 본 논문의 방법을 현재 주류인 Transformer 기반 아키텍처에 적용할 필요가 있습니다. 최신 연구에서 지적하는 LLM의 불안정성 문제를 해결하는 데 ASWA/NASWA의 변형이 도움이 될 수 있습니다.[2]

Transformer의 주의 메커니즘:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

여기서:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**다중 모달 모델**: 이미지-텍스트 정렬, 비전-언어 모델 등에서의 안정성 측정이 필요합니다.[10]

#### 8.2 이론적 심화

**2차 정보 활용**: 본 논문이 제시한 한계를 극복하기 위해 Hessian 정보를 활용한 고차 안정성 방법을 개발해야 합니다. Newton 방법 기반 접근:[1]

$$\theta_{t+1} = \theta_t - \alpha H^{-1}(\theta_t) \nabla L(\theta_t)$$

**일반화 바운드**: SWA의 일반화 특성에 대한 더 타이트한 이론적 경계 도출이 필요합니다:[8]

$$\mathcal{L}_{\text{gen}}(\bar{\theta}) \leq \mathcal{L}_{\text{train}}(\bar{\theta}) + \mathcal{O}(d \log(1/\delta) / n)$$

#### 8.3 실무적 응용

**임상/고위험 애플리케이션**: 의료 진단, 자율주행 등 고위험 도메인에서는 모델 해석의 안정성이 매우 중요하므로, ASWA/NASWA 기반 접근법의 실제 적용 사례 연구가 필요합니다.

**해석 가능성 보증**: ASWA/NASWA 적용 시 해석이 실제로 "신뢰할 수 있는" 해석인지를 보증하는 메커니즘 개발이 필요합니다:[5]

$$P(\text{Explanation}_i = \text{Explanation}_j | \text{ASWA/NASWA}) > 1 - \epsilon$$

#### 8.4 계산 효율성

**경량화 기법**: ASWA/NASWA의 계산 오버헤드를 최소화하면서 안정성 이득을 유지하는 방법을 개발해야 합니다.

시간 복잡도:
- 표준 학습: $$\mathcal{O}(T \cdot n \cdot d)$$
- ASWA/NASWA: $$\mathcal{O}(T \cdot n \cdot d + T \cdot n \cdot d)$$

여기서 $T$는 에포크, $n$은 배치 크기, $d$는 차원입니다.[1]

**분산 학습에서의 안정성**: 분산 학습 환경에서 서로 다른 기기/노드에서의 랜덤성 관리 방법을 개발해야 합니다.[11]

#### 8.5 동적 환경에서의 안정성

**연속 학습(Continual Learning)**: 새로운 데이터가 계속 추가되는 상황에서의 모델 안정성 유지 방법:[12]

$$L_{\text{continual}} = L_{\text{current}} + \lambda \cdot \text{EWC}(\theta)$$

여기서 EWC는 Elastic Weight Consolidation입니다.[12]

**온라인 학습**: 스트리밍 데이터에서의 예측 및 해석 안정성:[13]

$$\theta_{t+1} = \theta_t - \alpha \nabla L(x_t, y_t; \theta_t)$$

### 결론

"On Model Stability as a Function of Random Seed"는 **심층 신경망의 비결정적 특성**이 예측 성능뿐만 아니라 모델 해석까지도 심각하게 영향을 미친다는 중요한 통찰을 제공했습니다. ASWA와 NASWA 방법의 제안은 실용적이면서도 효과적인 해결책을 제시했으며, 이는 이후 모델 안정성 연구의 기초가 되었습니다. 

핵심 결과를 요약하면:[1]
- 표준 모델 대비 **72% 안정성 개선** (평균)
- DiabetesMIMIC에서 **89% 상대 개선**
- 해석 엔트로피 **60% 감소** (일부 데이터셋)
- 성능 유지/개선: $\Delta \text{Acc} \in [-0.2\%, +1.3\%]$

2024-2025년의 최신 연구들은 이 논문의 아이디어를 대규모 언어 모델, 그래프 신경망, 컴퓨터 비전 등 다양한 도메인으로 확장하고 있으며, 특히 해석 가능성과 안정성의 관계를 더욱 정교하게 분석하고 있습니다. 향후 연구는 현대적 신경망 아키텍처에 이 원리들을 적용하고, 이론적 이해를 심화시키며, 고위험 도메인에서의 실무적 가치를 입증하는 방향으로 발전할 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/292780b3-86e9-4e0c-a643-5dd44047a958/1909.10447v1.pdf)
[2](https://arxiv.org/pdf/2503.07329.pdf)
[3](https://arxiv.org/pdf/2304.06715.pdf)
[4](http://papers.neurips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf)
[5](https://www.tandfonline.com/doi/full/10.1080/08839514.2025.2515062)
[6](https://papers.nips.cc/paper_files/paper/2024/file/3f0c8c5dc6b16e601b78e164a70d68a2-Paper-Conference.pdf)
[7](https://arxiv.org/pdf/2312.01082.pdf)
[8](https://openreview.net/forum?id=XwVkqvyziD)
[9](https://openreview.net/pdf?id=nxumYwxJPB)
[10](https://arxiv.org/html/2405.01524v3)
[11](https://arxiv.org/html/2410.23495v1)
[12](https://arxiv.org/html/2406.06811v2)
[13](https://arxiv.org/pdf/2403.19871.pdf)
