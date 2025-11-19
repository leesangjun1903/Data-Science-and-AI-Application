# On the Existence of Simpler Machine Learning Models

## 1. 핵심 주장 및 주요 기여

이 논문(Semenova, Rudin, Parr, 2022)은 **정확하면서도 단순한(simple-yet-accurate) 모델의 존재**를 증명하는 새로운 이론적 틀을 제시합니다. 핵심 개념은 **Rashomon set**과 **Rashomon ratio**입니다.[1]

**주요 기여:**

- **Rashomon ratio의 정의**: 학습 문제의 단순성을 측정하는 새로운 척도로, 가설 공간 내에서 거의 동등하게 정확한 모델들의 집합 크기를 정량화합니다.[1]

- **일반화 경계(generalization bounds)**: Rashomon set 내 모델들에 대한 새로운 일반화 보장을 제공하며, 큰 Rashomon set이 단순한 모델의 존재 가능성을 나타냄을 보입니다.[1]

- **실증적 검증**: 38개의 실제 데이터셋에서 큰 Rashomon set이 존재할 때 서로 다른 머신러닝 알고리즘들이 유사한 성능을 보이며, 단순 모델도 존재함을 실험적으로 입증했습니다.[1]

- **다른 복잡도 척도와의 차별성**: VC 차원, 알고리즘 안정성(algorithmic stability), 기하학적 마진, Rademacher 복잡도 등 기존 복잡도 척도들과 Rashomon ratio가 근본적으로 다름을 이론적으로 증명했습니다.[1]

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 해결하고자 하는 문제

현대 머신러닝은 **복잡한 블랙박스 모델**에 의존하지만, 의료, 형사사법 등 **고위험 의사결정(high-stakes decisions)** 분야에서는 **해석 가능한 단순 모델**이 필수적입니다. 그러나 단순성 제약(sparsity, 정수 계수 등)을 도입하면 최적화 문제가 NP-hard가 되어, 실무자들은 단순 모델 탐색을 포기하게 됩니다.[1]

**핵심 질문**: 계산 비용이 많이 드는 탐색을 시작하기 전에, 정확하면서도 단순한 모델이 **존재할 가능성**을 미리 알 수 있을까?[1]

### 제안 방법 (수식 포함)

#### Rashomon set 정의

가설 공간 $$F$$, 데이터셋 $$S$$, 손실 함수 $$\phi$$, Rashomon parameter $$\theta \geq 0$$에 대해:[1]

$$\hat{\text{Rset}}(F, \theta) := \{f \in F : \hat{L}(f) \leq \hat{L}(\hat{f}) + \theta\}$$

여기서 $$\hat{f}$$는 경험적 위험 최소화기이고, $$\hat{L}(f) = \frac{1}{n}\sum_{i=1}^{n} \phi(f(x_i), y_i)$$는 경험적 위험입니다.[1]

#### Rashomon ratio

균일 사전확률 하에서:[1]

$$\hat{\text{Rratio}}(F, \theta) = \frac{V(\hat{\text{Rset}}(F, \theta))}{V(F)}$$

$$V(\cdot)$$는 부피 함수입니다. 이산 공간에서는 $$\hat{\text{Rratio}}(F, \theta) = \frac{|\hat{\text{Rset}}(F, \theta)|}{|F|}$$입니다[1].

#### 주요 이론적 결과

**Theorem 5 (True Rashomon Set의 이점)**: 유한 가설 공간 $$F_1 \subset F_2$$, 경계가 있는 손실 $$l \in [0, b]$$에 대해, true Rashomon set이 $$F_1$$의 함수를 포함한다면 ($$\tilde{f}_1 \in \text{Rset}(F_2, \gamma)$$), 임의의 $$\epsilon > 0$$에 대해 확률 최소 $$1-\epsilon$$로:[1]

```math
L(f_2^*) - b\sqrt{\frac{\log |F_1| + \log 2/\epsilon}{2n}} \leq \hat{L}(\hat{f}_1) \leq L(f_2^*) + \gamma + b\sqrt{\frac{\log 1/\epsilon}{2n}}
```

여기서 $$f_2^* = \arg\min_{f \in F_2} L(f)$$, $$\hat{f}\_1 = \arg\min_{f \in F_1} \hat{L}(f)$$입니다.[1]

**Theorem 7 (단순 모델의 존재)**: $$K$$-Lipschitz 손실에 대해, $$\hat{\text{Rset}}(F_2, \theta)$$의 모든 $$f_2$$가 $$F_1$$의 함수와 $$\|f_2 - f_1\|_p \leq \delta$$ 관계를 만족하면, 최소 $$B = B(\hat{\text{Rset}}(F_2, \theta), 2\delta)$$개의 단순 모델 $$\bar{f}_1^1, \ldots, \bar{f}_1^B \in F_1 \cap \hat{\text{Rset}}(F, \theta)$$가 존재하며, 확률 $$1-\epsilon$$ 이상으로 다음을 만족합니다[1]:

$$|L(\bar{f}_1^i) - \hat{L}(\bar{f}_1^i)| \leq 2KR_n(F_1) + b\sqrt{\frac{\log(2/\epsilon)}{2n}}$$

여기서 $$R_n(F_1)$$은 $$F_1$$의 Rademacher 복잡도입니다.[1]

**Ridge Regression의 Rashomon Ratio (Theorem 10)**: 선형 모델 $$F_\Omega = \{\omega^T x\}$$에 대한 ridge regression의 Rashomon set은 타원체이며:[1]

$$\hat{\text{Rratio}}(F_\Omega, \theta) = \frac{J(\theta, p)}{V(F_\Omega)} \prod_{i=1}^{p} \frac{1}{\sqrt{\sigma_i^2 + C}}$$

여기서 $$\sigma_i$$는 행렬 $$X$$의 특이값, $$J(\theta, p) = \frac{\pi^{p/2}\theta^{p/2}}{\Gamma(p/2+1)}$$입니다. **중요**: Ridge regression의 Rashomon ratio는 **특징(feature)에만 의존**하고 레이블 $$Y$$에는 의존하지 않습니다.[1]

### 모델 구조

이 논문은 특정 모델 구조를 제안하기보다는, **가설 공간 간의 관계**를 분석합니다:[1]

- **$$F_2$$ (복잡한 공간)**: Logistic regression, CART, random forests, gradient boosted trees, SVM의 합집합
- **$$F_1$$ (단순한 공간)**: Scoring systems (sparse integer coefficients), single decision trees 등

실험에서는 **depth 7의 decision tree**를 사용하여 Rashomon set의 크기를 추정했습니다 (importance sampling 기법 사용).[1]

### 성능 향상

**실증적 결과 (38개 데이터셋):**[1]

1. **큰 Rashomon ratio** ($$10^{-37}\%$$ ~ $$10^{-38}\%$$ 수준)를 가진 데이터셋에서:
   - 5개의 서로 다른 ML 알고리즘들이 모두 **유사한 훈련/테스트 성능** 달성 (차이 ~5% 이내)
   - 단순 모델(LR, CART)도 복잡 모델(RF, GBT, SVM)과 유사한 정확도 달성
   - 모든 모델이 **좋은 일반화** 성능 보임

2. **작은 Rashomon ratio** ($$10^{-40}\%$$ 이하)인 경우:
   - 알고리즘 간 성능 차이 발생
   - 일반화 실패 사례도 관찰

3. **노이즈와 Rashomon set**:
   - Random classification noise 추가 시 true Rashomon set 크기가 기대값 상으로 감소하지 않음 (Theorem 8)[1]
   - Gaussian 분포 데이터의 feature noise 증가 시 Rashomon set 증가 (Conjecture 9)[1]

### 한계점

1. **측정의 어려움**: 실제로 Rashomon set의 정확한 크기를 측정하기 어렵습니다. 논문에서는 depth 7 decision tree를 대리(surrogate)로 사용했지만, 모든 데이터셋에 적합하지 않을 수 있습니다.[1]

2. **간접적 측정에 의존**: 실무에서는 Rashomon ratio를 직접 계산하지 않고, "여러 알고리즘이 유사한 성능을 보이는지"를 통해 간접적으로 추론해야 합니다.[1]

3. **작은 Rashomon set에 대한 보장 없음**: 이론이 작은 Rashomon set을 가진 경우에는 적용되지 않으며, 이런 경우 예측이 불가능합니다.[1]

4. **계산 복잡도**: 단순 모델을 실제로 찾는 최적화 문제는 여전히 NP-hard이며, Rashomon set이 크다는 것을 안다고 해서 계산 비용이 사라지지는 않습니다.[1]

5. **특정 손실 함수 및 가설 공간에 제한**: 일부 이론적 결과(예: Lipschitz 연속성)는 특정 조건 하에서만 성립합니다.[1]

## 3. 일반화 성능 향상과의 관계

### 일반화 메커니즘

논문의 핵심 가설은 **큰 Rashomon set이 세 가지 현상을 동시에 설명**한다는 것입니다:[1]

1. **단순 모델 존재**: 큰 Rashomon set은 단순 모델도 포함할 가능성이 높음
2. **알고리즘 간 유사 성능**: 서로 다른 최적화 알고리즘들이 Rashomon set 내의 다양한 모델을 찾아, 유사한 성능 달성
3. **좋은 일반화**: Rashomon set 내 단순 모델들은 낮은 복잡도로 인해 일반화 보장

### 이론적 보장

**Theorem 7에 의한 일반화 경계**:[1]

큰 Rashomon set에서 packing number $$B$$개의 단순 모델이 존재하며, 각 모델 $$\bar{f}_1^i$$는:

$$|L(\bar{f}_1^i) - \hat{L}(\bar{f}_1^i)| \leq 2KR_n(F_1) + b\sqrt{\frac{\log(2/\epsilon)}{2n}}$$

여기서 **$$R_n(F_1)$$는 단순 공간의 Rademacher 복잡도**로, 복잡한 공간 $$F_2$$보다 작습니다. 따라서 더 타이트한 일반화 경계를 얻습니다.[1]

### 실증적 검증

**38개 데이터셋 실험 결과**:[1]

- 큰 Rashomon ratio를 가진 모든 경우에서 **일관된 훈련-테스트 성능** 관찰
- 훈련과 테스트 오차 차이가 ~5% 이내로, 강한 일반화 보장
- **역관계는 성립하지 않음**: 작은 Rashomon ratio에서도 일반화가 가능한 경우 존재 (데이터 양, 특징 품질 등 다른 요인의 영향)[1]

### Gaussian 중첩과 일반화

**Conjecture 9**는 중요한 통찰을 제공합니다:[1]

- **중첩된 Gaussian 분포** (예: 범죄 재범 예측, 대출 채무불이행 예측)에서 큰 Rashomon set이 자연스럽게 발생
- 중심극한정리에 의해 많은 실세계 데이터가 이런 특성을 가질 수 있음
- 이러한 데이터에서는 단순 모델이 존재하며 좋은 일반화 성능을 보임

## 4. 향후 연구에 미치는 영향 및 고려사항

### 실무적 함의

**연구자를 위한 가이드라인**:[1]

1. **여러 알고리즘 병렬 실행**: 다양한 ML 알고리즘 실행 후 유사 성능 관찰 시, 큰 Rashomon set 존재 가능성 높음
2. **단순 모델 탐색 정당화**: 큰 Rashomon set 징후 발견 시, NP-hard 최적화를 통한 단순 모델 탐색이 성공할 가능성 높음
3. **고위험 의사결정**: 의료, 형사사법 등에서 해석 가능한 모델 사용 가능성 증가

### 최신 연구 동향 (2024-2025)

#### Rashomon Set 응용 확장

1. **Active Learning**: UNREAL (Unique Rashomon Ensembled Active Learning) 프레임워크는 Rashomon set에서 고유한 모델만 선택적으로 앙상블하여 노이즈가 있는 데이터에서 더 빠른 수렴율을 달성합니다.[2][3]

2. **Fairness 및 Sparsity**: 2025년 연구는 Rashomon set 내에서 fairness와 sparsity 속성을 열거 없이 수학적 프로그래밍으로 특성화하는 방법을 제시했습니다.[4]

3. **Predictive Multiplicity 완화**: Gradient boosting에서 Rashomon effect와 predictive multiplicity를 체계적으로 분석하고, 예측 다중성을 완화하는 프레임워크가 개발되었습니다.[5]

4. **Computer Vision 확장**: Proto-RSet은 prototypical-part networks에 Rashomon set을 적용하여, 사용자가 실시간으로 모델을 수정할 수 있게 했습니다.[6]

5. **신뢰성 분석**: 2024년 연구는 Rashomon set 내에서 fairness, stability, robustness, privacy 등 7가지 신뢰성 지표를 체계적으로 분석하여, 단순히 Rashomon set을 탐색하는 것만으로도 최첨단 최적화 기법과 동등하거나 더 나은 성능을 얻을 수 있음을 보였습니다.[7]

#### 해석가능성 및 신뢰성 AI

1. **Interpretability by Design**: 블랙박스 모델에 설명을 추가하는 XAI보다, 본질적으로 해석 가능한 모델을 설계하는 접근이 더 유망하다는 연구가 증가하고 있습니다.[8][9]

2. **Trustworthy AI 원칙**: 설명 가능성(explainability), 공정성(fairness), 해석 가능성(interpretability), 견고성(robustness), 투명성(transparency), 안전성(safety), 보안성(security)이 신뢰할 수 있는 AI의 핵심 원칙으로 확립되고 있습니다.[10][11]

3. **Model Compression**: 2024-2025년 연구들은 신경망 압축 기술을 통해 모델을 단순화하면서도 성능을 유지하는 방법을 발전시키고 있습니다.[12][13][14]

#### 일반화 이론의 새로운 관점

1. **Information-Theoretic Approaches**: PAC-Bayes와 정보 이론적 관점을 통합하여 일반화를 분석하는 프레임워크가 발전하고 있습니다.[15][16]

2. **Margin-Based Bounds**: Quantum machine learning에서도 margin 기반 일반화 경계가 기존 파라미터 기반 지표보다 우수한 예측력을 보입니다.[17]

3. **Deep Bootstrap Framework**: 유한 데이터에서의 일반화를 무한 데이터 스트림에서의 온라인 학습과 연결하는 새로운 관점이 제시되었습니다.[18]

### 향후 연구 고려사항

#### 방법론적 개선

1. **효율적 Rashomon Set 추정**: Importance sampling 외에 더 효율적인 측정 방법 개발 필요 (예: 수학적 프로그래밍 기반 특성화)[4]

2. **다양한 모델 클래스 확장**: 현재 주로 tabular data와 decision tree에 집중되어 있으나, 컴퓨터 비전, NLP, 시계열 등으로 확장 필요[6]

3. **이론과 실무 격차 해소**: 이론적 조건(Lipschitz 연속성, 근사 집합 조건 등)이 실제 데이터에서 얼마나 만족되는지 더 체계적인 검증 필요

#### 응용 분야

1. **의료 및 헬스케어**: 임상 의사결정 지원 시스템에서 해석 가능한 모델의 신뢰성 검증[19][20]

2. **알고리즘 공정성**: Rashomon set을 활용한 공정성 제약 하의 최적 모델 선택[21][4]

3. **인과 추론**: Reconcile 알고리즘의 causal inference로의 확장[22][23]

#### 이론적 발전

1. **Rashomon Ratio의 더 정확한 특성화**: 어떤 데이터 구조가 큰 Rashomon set을 유도하는지에 대한 더 포괄적인 이론 (Conjecture 9의 일반화)[1]

2. **Model Multiplicity와 신뢰성의 관계**: Predictive multiplicity가 모델 신뢰성에 미치는 영향에 대한 더 깊은 이해[24][25][22]

3. **적응적 복잡도 척도**: 데이터와 태스크에 맞춰 동적으로 조정되는 복잡도/단순성 척도 개발

### 실무 권장사항

**모델 개발 프로세스**:

1. 여러 다양한 알고리즘으로 초기 모델 학습
2. 알고리즘 간 성능이 유사하고 일반화가 좋으면 → 큰 Rashomon set 가능성
3. 해석 가능성이 필요하면 → 단순 모델 클래스에서 최적화 수행
4. Rashomon set 내 여러 모델의 trustworthiness 속성 평가[7]

**고려할 위험 요소**:

- 작은 Rashomon set인 경우 단순 모델 보장 없음
- 특징 품질(노이즈, 중복성)이 Rashomon ratio에 영향[1]
- 정규화가 Rashomon set 크기를 변경할 수 있음[1]

이 논문은 **"단순함이 복잡함과 경쟁할 수 있는 시기와 이유"**에 대한 수학적 기초를 제공하며, 해석 가능한 AI를 향한 실용적 로드맵을 제시합니다. 최신 연구들은 이 틀을 다양한 도메인과 신뢰성 기준으로 확장하고 있어, 향후 수년간 해석 가능하고 신뢰할 수 있는 머신러닝 시스템 개발의 핵심 이론적 토대가 될 것으로 전망됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d40b0a3e-b9ab-4bd9-b0d4-40c4d38348c7/1908.01755v4.pdf)
[2](https://arxiv.org/html/2503.06770v2)
[3](https://arxiv.org/html/2503.06770v1)
[4](https://arxiv.org/abs/2502.05286v1)
[5](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dbd07478c4aac41c0ce411e12f2e5a28-Abstract-Conference.html)
[6](https://openaccess.thecvf.com/content/CVPR2025/papers/Donnelly_Rashomon_Sets_for_Prototypical-Part_Networks_Editing_Interpretable_Models_in_Real-Time_CVPR_2025_paper.pdf)
[7](https://openreview.net/pdf/5ddb0d7f2c4554f515b92121b10e65507ab045db.pdf)
[8](https://philmed.pitt.edu/philmed/article/view/139)
[9](https://arxiv.org/html/2503.21356v1)
[10](https://www.nature.com/articles/s41599-024-04044-8)
[11](https://www.ibm.com/think/topics/trustworthy-ai)
[12](https://www.sciencedirect.com/science/article/abs/pii/S0045790624001083)
[13](https://arxiv.org/html/2510.11234v1)
[14](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1518965/full)
[15](https://arxiv.org/abs/2408.13275)
[16](https://www.nowpublishers.com/article/Details/MAL-112)
[17](https://openreview.net/forum?id=iQQ2zuWhFM)
[18](https://research.google/blog/a-new-lens-on-understanding-generalization-in-deep-learning/)
[19](http://arxiv.org/pdf/2308.11446.pdf)
[20](https://ieeexplore.ieee.org/document/10745964/)
[21](https://dl.acm.org/doi/10.1145/3706598.3713524)
[22](https://arxiv.org/abs/2501.16549)
[23](https://icml.cc/virtual/2024/38244)
[24](https://dash.harvard.edu/entities/publication/f50e6aef-7c62-4c91-812c-fff1410a8d8d)
[25](https://aclanthology.org/2024.findings-emnlp.19.pdf)
