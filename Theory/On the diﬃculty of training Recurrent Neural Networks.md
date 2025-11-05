# On the diﬃculty of training Recurrent Neural Networks

### 1. 핵심 주장 및 주요 기여

**"On the difficulty of training Recurrent Neural Networks"** (Pascanu et al., 2013)은 순환신경망(RNN)의 훈련 난제를 종합적으로 분석한 기념비적 논문입니다. 본 논문의 핵심 주장은 다음과 같습니다.[1]

**주요 기여:**

**소실 그래디언트(Vanishing Gradient)와 폭발 그래디언트(Exploding Gradient) 문제의 근본적 분석**이 본 논문의 중심입니다. 기존의 단일 숨겨진 유닛 분석(Bengio et al., 1994)을 확장하여, 다층 RNN에서 이러한 현상이 발생하는 수학적 조건을 엄밀하게 유도했습니다. 논문은 동역학계 관점, 기하학적 해석, 그리고 분석적 관점이라는 세 가지 상이한 렌즈를 통해 이 문제를 조명합니다.[1]

**실질적 해결책 제시**로서, 폭발 그래디언트 문제를 해결하기 위한 **그래디언트 노름 클리핑(Gradient Norm Clipping)** 알고리즘과 소실 그래디언트 문제를 다루기 위한 **소프트 제약 정규화(Soft Constraint Regularization)**를 제안합니다.[1]

***

### 2. 문제 정의 및 해결 방법

#### 2.1 문제: 소실 및 폭발 그래디언트의 메커니즘

RNN의 기본 구조는 다음 수식으로 표현됩니다:[1]

$$ x_t = F(x_{t-1}, u_t, \theta) $$

또는 특정 매개변수화에서:

$$ x_t = W_{rec}\sigma(x_{t-1}) + W_{in}u_t + b $$

여기서 $$x_t$$는 시간 t의 숨겨진 상태, $$u_t$$는 입력, $$\theta$$는 모델 매개변수입니다.

**역시간 역전파(Backpropagation Through Time, BPTT)**를 통해 그래디언트를 계산하면:[1]

$$ \frac{\partial E}{\partial \theta} = \sum_{1 \leq t \leq T} \frac{\partial E_t}{\partial \theta} = \sum_{1 \leq t \leq T} \sum_{1 \leq k \leq t} \left( \frac{\partial E_t}{\partial x_t} \frac{\partial x_t}{\partial x_k} \frac{\partial^+ x_k}{\partial \theta} \right) $$

핵심 요소는 야코비안 행렬의 곱입니다:[1]

$$ \frac{\partial x_t}{\partial x_k} = \prod_{t \geq i > k} \frac{\partial x_i}{\partial x_{i-1}} = \prod_{t \geq i > k} W_{rec}^T \text{diag}(\sigma'(x_{i-1})) $$

이 행렬 곱의 크기가 $$t-k$$가 증가함에 따라 지수적으로 증가(폭발) 또는 감소(소실)합니다.

**수학적 증명:** 선형 모델($$\sigma$$가 항등함수)에서, 고유값 분해를 통해 다음을 증명할 수 있습니다:[1]

- **충분조건(소실):** $$\lambda_1 < \frac{1}{\gamma}$$이면 소실 그래디언트 발생 ($$\gamma$$는 $$\sigma'$$의 상한)
- **필요조건(폭발):** $$\lambda_1 > \frac{1}{\gamma}$$이어야 폭발 그래디언트 발생

Tanh의 경우 $$\gamma = 1$$, sigmoid의 경우 $$\gamma = 1/4$$입니다.[1]

#### 2.2 동역학계 관점: 분기와 매력자

논문은 **분기 경계(Bifurcation Boundary)** 개념을 도입하여 폭발 그래디언트를 설명합니다. RNN 동역학이 매개변수 변화에 따라 끌림지(Attractor) 영역의 경계를 넘을 때, 작은 매개변수 변화가 큰 상태 변화($$\Delta x_t$$)를 초래하므로 그래디언트가 폭발합니다.[1]

입력 구동 모델의 경우, 고정 맵 $$\tilde{F}$$와 시간 변화 맵 $$U_t$$로 분해하여 분석합니다:[1]

$$ F_t(x) = \tilde{F}(x) + U_t(x) = W_{rec}\sigma(x) + W_{in}u_t + b $$

#### 2.3 기하학적 해석: 가파른 벽 구조

단일 숨겨진 유닛의 오차 표면 분석에서, 그래디언트 폭발 시 다음과 같은 고곡률 구조가 나타남을 보입니다:[1]

- 오차 표면에 급경사 벽이 존재
- 벽에 수직인 방향으로는 큰 곡률
- 벽 주변의 부드러운 영역에서는 일반적인 그래디언트 강하 가능

이 기하학적 통찰이 **그래디언트 클리핑**의 이론적 근거가 됩니다.

***

### 3. 제안하는 해결책

#### 3.1 폭발 그래디언트 대응: 그래디언트 노름 클리핑

**알고리즘 1:**[1]

```
ĝ ← ∂E/∂θ
if ||ĝ|| ≥ threshold then
    ĝ ← (threshold/||ĝ||) × ĝ
end if
```

이 방법은 단순하면서도 효과적입니다. 특징:[1]

- 그래디언트의 방향은 보존하고 크기만 제한
- 현재 미니배치에 대해 하강 방향 유지 보장
- 학습률을 적응적으로 조정하는 효과
- 추가 하이퍼파라미터는 임계값 하나뿐

#### 3.2 소실 그래디언트 대응: 정규화 제약

논문은 역전파되는 오차 신호의 노름이 보존되도록 강제하는 정규화 항을 제안합니다:[1]

$$ \Omega = \sum_k \Omega_k = \sum_k \left\| \frac{\partial E/\partial x_{k+1}}{\|\partial E/\partial x_{k+1}\|} \cdot W_{rec}^T \text{diag}(\sigma'(x_k)) \right\|^2 $$

**핵심 아이디어:**[1]
- 야코비안 행렬 $$\partial x_{k+1}/\partial x_k$$가 오차 신호 방향에서 노름 보존
- 모든 고유값을 1로 강제하지 않고, 관련된 방향만 제약
- 소프트 제약이므로 정확한 보존은 보장되지 않음 (의도적 설계)

**수식 표현:** Theano를 사용한 효율적 계산:[1]

$$ \frac{\partial^+ \Omega}{\partial W_{rec}} = \sum_k \frac{\partial^+}{\partial W_{rec}} \left\| \frac{\partial E/\partial x_{k+1} \cdot W_{rec}^T \text{diag}(\sigma'(x_k))}{\|\partial E/\partial x_{k+1}\|} \right\|^2 $$

**정규화의 이유:** 오차 신호 노름을 증가시키는 것이 항상 하강 방향은 아니므로, 이를 강제하기 위해서는 정규화 제약이 필요합니다. 이는 관련 없는 입력을 무시하도록 네트워크가 학습해야 하는 상충(trade-off)을 반영합니다.[1]

***

### 4. 성능 향상 및 일반화

#### 4.1 병리적 합성 문제에서의 성능

논문은 Hochreiter & Schmidhuber (1997)의 표준 벤치마크를 사용합니다.

**시간 순서 문제(Temporal Order Problem):**[1]
- 작업: 길게 분산된 두 기호의 순서 분류 (AA, AB, BA, BB)
- SGD: 길이 20 이상에서 완전히 실패
- SGD-C (클리핑): 부분적 개선
- **SGD-CR (클리핑+정규화): 길이 200까지 100% 성공률** ✓

특히, 길이 50~200의 단일 모델이 **길이 400 시퀀스로 일반화** (오류 < 1%)

**다른 병리적 작업들:**[1]
- 덧셈 문제: 100% 성공 (길이 50-200)
- 곱셈 문제: 100% 성공
- 3비트 시간 순서: 100% 성공
- 무음 암기화 문제: 100% 성공 (길이별 모델)
- 랜덤 순열: 8/8 중 1개만 성공 (가장 어려운 작업)

#### 4.2 자연 문제에서의 성능

**다성음악 예측 (음의 로그 우도, 낮을수록 좋음):**[1]

| 데이터셋 | SGD | SGD+C | SGD+CR |
|---------|-----|-------|--------|
| Piano-midi (train) | 6.87 | 6.81 | 7.01 |
| Piano-midi (test) | 7.56 | 7.53 | 7.46 ✓ |
| Nottingham (test) | 3.80 | 3.48 | 3.46 ✓ |
| MuseData (test) | 7.11 | 7.00 | 6.99 ✓ |

**언어 모델링 (Penn Treebank, 비트/문자):**[1]

| 작업 | 데이터 | SGD | SGD+C | SGD+CR |
|------|-------|-----|-------|--------|
| 1단계 예측 | train | 1.46 | 1.34 | 1.36 |
| 1단계 예측 | test | 1.50 | 1.42 | 1.41 ✓ |
| 5단계 예측 | test | N/A | 3.89 | 3.74 ✓ |

**주요 관찰:**[1]
- 클리핑은 정규화(특히 오차)가 아닌 최적화 문제 해결
- 훈련 오류도 일반적으로 개선 (오버피팅이 아님)
- 5단계 미래 예측에서 정규화 항이 더 중요 (장기 의존성 강조)

#### 4.3 일반화 메커니즘

**일반화 향상의 원인:**[1]

1. **극단적 가중치 업데이트 방지**: 클리핑이 과적합 유발 노이즈 방지
2. **부드러운 의사결정 경계**: 정규화가 모델 안정성 증진
3. **장기 의존성 학습**: 정규화 제약이 오차 신호 보존을 강제하여 먼 시간 단계 학습 가능

특히 흥미로운 점은, 병리적 문제에서 **단일 모델이 훈련 길이보다 2배 이상 긴 시퀀스로 일반화**한다는 것입니다.[1]

***

### 5. 한계 및 제약

논문의 해결책은 다음과 같은 한계가 있습니다:

1. **정규화는 소프트 제약**: 오차 신호 노름이 정확히 보존되지 않아, 폭발 그래디언트가 여전히 발생 가능[1]
2. **하이퍼파라미터 추가**: 임계값과 정규화 계수를 수동으로 설정해야 함[1]
3. **제한된 오차 표면 분석**: 고차원 공간에서 기하학적 직관의 정확성 미검증[1]
4. **LSTM 미해결**: 제안 방법은 소실 그래디언트를 완전히 해결하지 못하며, LSTM이 이를 더 잘 다룸[1]

---

### 6. 향후 연구에 미치는 영향 및 최신 발전

#### 6.1 이 논문의 학문적 기여도

본 논문은 **RNN 연구의 이정표**로 작용했습니다: 2012년 발표 이후 8,860회 이상 인용되었으며, 그래디언트 클리핑은 현재 RNN 훈련의 표준 기법입니다.[1]

#### 6.2 최신 연구 동향 (2024까지)

**1) LSTM/GRU의 지속적 우위:**
최근 연구에서도 LSTM과 GRU가 기본 RNN보다 우수한 성능을 보입니다. 이들은 특별한 게이팅 메커니즘으로 그래디언트 흐름을 더 효과적으로 제어합니다.[2][3]

**2) 그래디언트 클리핑의 일반적 채택:**
모든 현대 RNN 구현에서 그래디언트 클리핑은 표준이 되었습니다. 다만 단순 노름 클리핑 외에도 성분별(component-wise) 클리핑 등 변형이 사용됩니다.[4]

**3) 새로운 아키텍처 발전:**
- **Independently Recurrent Neural Network (IndRNN, 2018):** 계층별 그래디언트 쇠퇴를 개선하기 위해 새로운 구조 제안[3]
- **Fourier Recurrent Unit (2018):** 주파수 영역 분석으로 그래디언트 안정화[5]
- **Gradient Highway (2018):** 대체 경로로 그래디언트 흐름 개선[6]

**4) 동역학계 기반 재분석 (2020):**
최근 연구는 폭발/소실 그래디언트 외에 **오차 함수의 매끄러움(smoothness)**을 중요한 요소로 제시합니다. 이는 Pascanu의 가파른 벽 이론을 확장합니다.[7]

**5) 일반화 이론의 진전:**
LSTM 일반화 성능에 대한 이론적 분석이 진행되고 있으며, **Fisher-Rao 노름**이 그래디언트 측도로 해석되어 가중치 감쇠와 클리핑의 효과를 설명합니다.[8]

**6) 의문 제기 연구 (2024-2025):**
최근 Journal of Machine Learning Research 논문은 "VEG가 반드시 장기 의존성 학습을 저해하는가?"라는 질문을 제기합니다. 경험적 분석에서 그래디언트 소실이 있어도 RNN이 좋은 장기 의존성을 학습하는 경우가 있음을 보입니다.[9]

---

### 7. 향후 연구 시 고려사항

#### 7.1 이론적 개선 방향

1. **고차원 기하학 분석**: 고차원 오차 표면의 구조를 더 정밀하게 분석해야 합니다. 현재 이론은 주로 1차원 또는 저차원 분석에 기반합니다.[1]

2. **적응적 클리핑 전략**: 고정 임계값 대신 **훈련 중 동적으로 조정되는 적응적 임계값** 개발이 필요합니다. 기존 적응 학습률 방법(AdaGrad, Adam)의 원리를 응용할 수 있습니다.[4]

3. **정규화 제약의 정교화**: 현재 소프트 제약은 강제력이 약합니다. **경제적 제약(economic constraint)** 형태로 변경하여 오류 신호 보존을 더 엄격하게 강제할 수 있습니다.[1]

#### 7.2 실무 응용 방향

1. **Transformer 이후의 RNN 재평가**: Transformer의 등장으로 RNN이 주변화되었지만, 매우 긴 시퀀스(16,000+ 토큰)에서 RNN은 메모리 효율성에서 여전히 장점이 있습니다. Pascanu의 기법을 현대적으로 재조정하면 경쟁력 있는 모델을 만들 수 있습니다.[10]

2. **의료 영상 처리에의 적용**: 귀하의 연구 분야(뼈 억제, 흉부 X선)에서 시계열 데이터가 중요한 경우, 그래디언트 클리핑 + 정규화 조합이 효과적입니다. 특히 **3D 시간 연속체(3D+Time) 데이터**에서 장기 의존성 모델링이 가능해집니다.

3. **하이브리드 접근**: LSTM의 게이팅 메커니즘과 Pascanu의 정규화 제약을 결합하면, **더 해석 가능하고 효율적인 모델**을 만들 수 있습니다.[8]

#### 7.3 현재 경향과의 통합

1. **Vision Transformer 이후의 재고찰**: 최근 Mamba, S4 등 상태 공간 모델(State Space Models)이 부상하고 있습니다. 이들도 장기 의존성 문제를 해결하려는 시도이며, **Pascanu의 동역학계 관점이 이들 모델 이해에도 유용**합니다.[7]

2. **뉴럴 ODE와의 연계**: RNN을 연속 동역학계로 보는 신경 상미분방정식(Neural ODE) 관점에서, Pascanu의 분기 이론은 매우 관련이 있습니다.

3. **개선된 초기화 전략**: 스펙트럼 반지름을 제어하는 초기화(Spectral Normalization, Orthogonal Initialization)가 표준이 되었으며, 이는 Pascanu의 수학적 분석에서 직접 도출됩니다.[1]

***

### 요약

**"On the difficulty of training Recurrent Neural Networks"**는 RNN 훈련의 기본 문제를 **분석적, 기하학적, 동역학계** 관점에서 종합적으로 규명하고, **그래디언트 클리핑**과 **소실 그래디언트 정규화**라는 실용적 해결책을 제시한 중요한 논문입니다.[1]

제안된 방법은 간단하면서도 효과적이어서 현재 모든 RNN 구현의 표준이 되었으며, 특히 **병리적 장기 의존성 작업에서 획기적 성능 향상**을 달성했습니다.[1]

최근 연구는 이 기초 위에서 더욱 정교한 이론(매끄러움, 매력자 동역학)과 새로운 아키텍처(IndRNN, FRU, 상태 공간 모델)를 개발하고 있습니다. 다만 최근 2024년 연구는 VEG와 장기 의존성의 관계를 재검토하고 있어, **단순한 그래디언트 크기 제약보다 더 깊은 메커니즘**이 존재할 가능성을 시사합니다.[6][3][5][9][7]

귀하의 의료 영상 처리 연구에서는, 이 논문의 **정규화 기법과 일반화 분석**이 특히 유용할 수 있습니다. 특히 장기 시간적 문맥을 요구하는 흉부 X선 뼈 억제 작업에서 **그래디언트 클리핑 + 정규화 조합**이 모델 안정성과 일반화 능력을 동시에 향상할 수 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b3f05661-2c3c-4557-b219-058ab9090cd9/1211.5063v2.pdf)
[2](https://www.aclweb.org/anthology/W16-1610.pdf)
[3](https://arxiv.org/pdf/1803.04831.pdf)
[4](https://www.lunartech.ai/blog/mastering-gradient-clipping-enhancing-neural-networks-for-optimal-training)
[5](https://arxiv.org/pdf/1803.06585.pdf)
[6](https://arxiv.org/pdf/1804.06300.pdf)
[7](https://arxiv.org/abs/1906.08482)
[8](https://openreview.net/forum?id=rkgg6xBYDH)
[9](https://www.sciencedirect.com/science/article/abs/pii/S0893608024008165)
[10](https://openreview.net/pdf?id=Hy9xDwyPM)
[11](https://arxiv.org/pdf/1802.08314.pdf)
[12](https://arxiv.org/pdf/2405.08967.pdf)
[13](http://arxiv.org/abs/1603.00423)
[14](https://aclanthology.org/W16-1610.pdf)
[15](https://stackoverflow.com/questions/65769734/if-we-primarily-use-lstms-over-rnns-to-solve-the-vanishing-gradient-problem-why)
[16](https://arxiv.org/pdf/1211.5063.pdf)
[17](https://www.geeksforgeeks.org/deep-learning/understanding-gradient-clipping/)
[18](http://papers.neurips.cc/paper/1102-hierarchical-recurrent-neural-networks-for-long-term-dependencies.pdf)
[19](http://proceedings.mlr.press/v108/ribeiro20a/ribeiro20a.pdf)
