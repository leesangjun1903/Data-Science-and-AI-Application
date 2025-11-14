# Cluster Alignment with a Teacher for Unsupervised Domain Adaptation

### 1. 핵심 주장과 주요 기여

**핵심 주장**[1]

논문의 중심 주장은 기존 Unsupervised Domain Adaptation (UDA) 방법들이 **한계를 가지고 있다**는 점입니다. 기존 방법들은 주로 **주변분포(marginal distribution) 정렬에만 집중**하여 두 도메인 간의 세밀한 클래스-조건부 구조(class-conditional structure)를 무시합니다. 이는 특히 도메인이 다양한 모드(diverse modes)를 가지거나 클래스 불균형 비율이 다를 때 심각한 성능 저하를 초래합니다.[1]

**주요 기여**[1]

1. **판별적 클래스-조건부 구조의 활용**: CAT는 두 도메인에서 판별적 클래스-조건부 구조를 명시적으로 발견하고 활용하여 더 효과적인 적응을 실현합니다.

2. **기존 UDA 방법과의 호환성**: CAT는 기존의 주변분포 정렬 방법(예: RevGrad, JAN)과 결합 가능하며, 이들을 클러스터 기반 정렬로 편향시켜 성능을 크게 향상시킵니다.

3. **실증적 우수성**: 다양한 벤치마크(SVHN-MNIST-USPS, Office-31, ImageCLEF-DA)에서 최첨단 성능을 달성합니다.

***

### 2. 해결하는 문제와 제안 방법

**문제 정의**[1]

UDA 문제는 다음과 같이 정의됩니다. 라벨된 소스 데이터셋 $$X_s = \{x_i^s\}\_{i=1}^N$$ 과 라벨되지 않은 타겟 데이터셋 $$X_t = \{x_i^t\}\_{i=1}^M$$가 주어졌을 때, 타겟 도메인에서 우수한 성능을 보이는 분류기를 학습하는 것입니다. 두 도메인은 서로 다른 분포에서 나왔으므로 도메인 적응이 필요합니다.[1]

**이론적 배경**[1]

이론상 타겟 도메인의 기대 오차는 다음과 같이 상한이 설정됩니다:

$$\epsilon_t(h) \leq \epsilon_s(h) + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(s,t) + \epsilon_t(l_s, l_t) + \min_{\hat{h} \in \mathcal{H}}(\epsilon_s(\hat{h}, l_s) + \epsilon_t(\hat{h}, l_s))$$

여기서 $$d_{\mathcal{H}\Delta\mathcal{H}}(s,t)$$는 두 도메인 간 H-divergence이고, $$\epsilon_t(l_s, l_t)$$는 타겟 도메인의 라벨 함수 불일치입니다. 기존 방법들은 $$\epsilon_s(h)$$와 $$d_{\mathcal{H}\Delta\mathcal{H}}(s,t)$$를 최소화하지만, $$\epsilon_t(l_s, l_t)$$를 무시합니다.[1]

**CAT의 최적화 목표**[1]

CAT는 다음 최적화 문제를 풀이합니다:

$$\min_\theta L_y + \alpha(L_c + L_a)$$

여기서:
- $$L_y$$: 소스 도메인의 지도학습 손실
- $$L_c$$: 판별적 클러스터링 손실
- $$L_a$$: 클러스터 정렬 손실
- $$\alpha$$: 가중치 하이퍼파라미터

**3. 모델 구조와 핵심 모듈**

**3.1 Teacher-Student 패러다임**[1]

CAT는 teacher 모델(이전 student classifier의 implicit ensemble)을 도입하여 타겟 도메인의 pseudo-label을 생성합니다. 이는 자기-학습의 오류 증폭 문제를 완화합니다. Teacher는 다음과 같이 정의됩니다:

- **합성 방법**: 이전 predictions의 temporal ensemble 사용 (decay constant: 0.6)
- **또는**: dropout을 통한 다른 perturbation 하에서의 두 번 forward propagation

**3.2 판별적 클러스터링 손실 ($$L_c$$)**[1]

SNTG 손실에서 영감을 받은 클러스터링 손실:

$$L_c(X) = \frac{1}{|X|^2}\sum_{i=1}^{|X|}\sum_{j=1}^{|X|}\left[\delta_{ij}d(f(x_i), f(x_j)) + (1-\delta_{ij})\max(0, m - d(f(x_i), f(x_j)))\right]$$

여기서:
- $$d$$: 거리 함수 (예: 제곱 유클리드 거리)
- $$m$$: 마진 (synthetic 실험에서 30, 다른 실험에서 3)
- $$\delta_{ij}$$: 같은 클래스 여부 지시자 (소스는 실제 라벨, 타겟은 teacher 라벨 사용)

**동작 원리**: 같은 클래스의 특징들을 서로 가깝게 당기고, 다른 클래스의 특징들을 최소 거리 m만큼 멀리 밀어냅니다.[1]

**3.3 클러스터 정렬 손실 ($$L_a$$)**[1]

조건부 특징 매칭을 통한 클래스-조건부 정렬:

$$L_a(X_s, X_t) = \frac{1}{K}\sum_{k=1}^{K}\|\lambda_{s,k} - \lambda_{t,k}\|_2^2$$

여기서:

$$\lambda_{s,k} = \frac{1}{|X_{s,k}|}\sum_{x_i^s \in X_{s,k}}f(x_i^s), \quad \lambda_{t,k} = \frac{1}{|X_{t,k}|}\sum_{x_i^t \in X_{t,k}}f(x_i^t)$$

- $$X_{s,k}$$: 클래스 k의 소스 샘플 부분집합
- $$X_{t,k}$$: teacher가 클래스 k로 어노테이션한 타겟 샘플 부분집합

**동작 원리**: 각 클래스마다 소스와 타겟의 클러스터 중심(평균)을 정렬하여 클래스-조건부 분포를 일치시킵니다.[1]

**3.4 개선된 주변분포 정렬 (Confidence-Thresholding)**[1]

기존 방법과 결합할 때, CAT는 robust RevGrad (rRevGrad)를 제안합니다:

$$\min_\theta\max_\phi L_d(X_s, X_t) = \frac{1}{N}\sum_{i=1}^{N}\log c(f(x_i^s; \theta); \phi) + \frac{1}{\tilde{M}}\sum_{i=1}^{\tilde{M}}\log(1 - c(f(x_i^t; \theta); \phi))\gamma_i$$

여기서 $$\gamma_i$$는 teacher의 신뢰도가 임계값 p(0.9로 설정)을 초과하는 경우만 1입니다.[1]

***

### 4. 성능 향상 및 실증적 결과

**4.1 불균형 데이터셋 성능**[1]

합성된 불균형 SVHN-MNIST-USPS 데이터셋(소스: 10:1 불균형, 타겟: 1:10 역불균형)에서:

| 방법 | SVHN→MNIST | MNIST→USPS | USPS→MNIST |
|------|-----------|-----------|-----------|
| RevGrad | 27.4% | 26.7% | 17.9% |
| MSTN | 25.8% | 30.3% | 29.4% |
| **CAT** | **100.0%** | **99.9%** | **99.9%** |

CAT가 다른 방법들을 압도적으로 능가합니다.[1]

**4.2 표준 Digits 벤치마크**[1]

| 방법 | SVHN→MNIST | MNIST→USPS | USPS→MNIST | 평균 |
|------|-----------|-----------|-----------|------|
| RevGrad | 73.9% | 77.1% | 73.0% | - |
| rRevGrad+CAT | **98.8%** | **94.0%** | **96.0%** | - |
| MCD | 96.2% | 94.2% | 94.1% | - |
| MCD+CAT | 97.1% | **96.3%** | **95.2%** | - |

CAT는 기존 방법들을 안정적으로 개선합니다.[1]

**4.3 Office-31 벤치마크 (ResNet-50)**[1]

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A | 평균 |
|------|-----|-----|-----|-----|-----|-----|------|
| RevGrad | 82.0% | 96.9% | 99.1% | 79.4% | 68.2% | 67.4% | 82.2% |
| rRevGrad+CAT | **94.4%** | **98.0%** | **100%** | **90.8%** | **72.2%** | **70.2%** | **87.6%** |
| GenToAdapt | 89.5% | 97.9% | 99.8% | 87.7% | 72.8% | 71.4% | 86.5% |

어려운 적응 작업(A→D, D→A, W→A)에서 특히 큰 개선을 보입니다.[1]

**4.4 ImageCLEF-DA 벤치마크 (ResNet-50)**[1]

평균 정확도:
- RevGrad: 85.0%
- rRevGrad+CAT: **87.3%**
- JAN+CAT: 86.4%

작은 타겟 도메인(600개 이미지)에서도 효과적입니다.[1]

***

### 5. 일반화 성능 향상 메커니즘

**5.1 클러스터 구조 시각화**[1]

t-SNE 시각화를 통해 CAT가 생성한 특징 공간의 특성:
- **RevGrad**: 클래스 간 겹침이 많고 덜 구분됨
- **rRevGrad+CAT**: 밀집되고 분리된 판별적 클러스터 형성

이는 정량적 K-means 클러스터링 정확도로도 검증됩니다:
- SVHN→MNIST: RevGrad 88%, rRevGrad+CAT 99%
- DSLR→Amazon: RevGrad 64%, rRevGrad+CAT 70%[1]

**5.2 일반화 향상의 이유**

1. **클래스-조건부 정렬**: 각 클래스가 양쪽 도메인에서 일관된 기하학적 구조를 가지도록 강제하여, 소스에서 학습한 분류기가 타겟에서 더 잘 일반화됩니다.

2. **판별적 특징**: $$L_c$$는 클래스 내 응집력과 클래스 간 분리를 동시에 최적화하므로, 타겟 도메인에서 더 정확한 예측을 가능하게 합니다.

3. **Teacher 앙상블의 안정성**: Teacher는 여러 예측의 ensemble이므로 단일 샘플의 오류 pseudo-label에 덜 민감합니다.

4. **신뢰도 필터링**: Confidence-thresholding은 불확실한 샘플을 제외하여 잘못된 클러스터 할당을 방지합니다.

***

### 6. 논문의 한계

**6.1 Hyperparameter 민감도**[1]

- 마진 m이 task마다 다르게 설정 필요 (digits: 30, 다른 작업: 3)
- 신뢰도 임계값 p는 고정(0.9)이지만, 다른 settings에서는 튜닝이 필요할 수 있음

**6.2 Mini-batch 구현의 한계**[1]

- Mini-batch에서 특정 클래스가 없으면 해당 항을 제거하는 방식은 클래스 통계를 부정확하게 만들 수 있음
- 소규모 배치에서는 클러스터 중심 추정이 부정확할 수 있음

**6.3 계산 복잡도**[1]

- $$L_c$$는 O(|X|²) 복잡도를 가지므로, 대규모 배치에서는 계산 비용이 높음
- 논문에서는 0.05배 추가 시간만 언급하지만, 대규모 데이터셋에서는 더 클 수 있음

**6.4 Pseudo-label 품질 문제**[1]

- Early training에서 teacher가 아직 부정확할 수 있어, 타겟 샘플의 pseudo-label이 신뢰할 수 없음
- 클래스 불균형이 심한 경우, teacher도 편향된 pseudo-label을 생성할 수 있음

***

### 7. 현대 UDA 연구 동향과 CAT의 위치

**7.1 2024-2025년 최신 연구**[2][3][4][5]

**클러스터링과 Prototypes의 활용**: 최근 연구들은 CAT와 유사하게 클러스터 구조와 prototype-based 방법을 활용합니다. 예를 들어, PALA (Prototype-Anchored Learning)는 class-imbalanced graph domain adaptation을 위해 prototype을 사용하며, prototype contrastive learning으로 정렬을 수행합니다.[6]

**Foundation Models와 Pre-trained Networks의 활용**: CLIP, Vision Transformers 같은 대규모 사전학습 모델을 활용한 UDA 방법들이 등장했습니다. 이들은 이미 좋은 표현을 가지고 있으므로, 단순 adaptation이나 prompt learning으로 효과적인 domain transfer를 달성합니다.[5]

**Teacher-Student와 Self-Training의 정교화**: CdKD-TSML은 teacher-student 패러다임을 확장하여, mutual learning을 통해 두 네트워크가 서로 감독하는 방식을 제안했습니다. Cycle Self-Training (CST)는 pseudo-label의 신뢰성 문제를 해결하기 위해 forward와 reverse 단계를 반복합니다.[7][8]

**Multi-Source Adaptation**: MPA (Multi-Prompt Alignment)와 같은 방법들이 여러 소스 도메인을 동시에 처리하는 방향으로 발전했습니다.[4]

**7.2 CAT의 지속적 영향**

1. **이론적 기초 제공**: CAT는 클래스-조건부 정렬의 중요성을 명시적으로 보였으므로, 이후 연구들이 이 개념을 발전시켰습니다.

2. **Teacher-Student의 표준화**: CAT 이후, teacher-student 패러다임이 UDA와 source-free DA의 표준 방식이 되었습니다.

3. **Confidence-based Filtering의 광범위 채택**: 신뢰도 기반 필터링은 이후 많은 방법에서 채택되었으며, pseudo-label 정제의 표준 기법이 되었습니다.

***

### 8. 향후 연구 시 고려할 점

**8.1 Foundation Models와의 통합**[9][10][11][12]

현재 추세는 ImageNet 사전학습 모델에서 대규모 vision-language 모델(CLIP, BERT)로 전환되고 있습니다. CAT는 다음과 같은 방식으로 현대적 모델에 적응될 수 있습니다:

- CLIP의 vision-text alignment를 활용한 semantic clustering
- Vision Transformer의 attention mechanism을 이용한 adaptive prototype selection
- Multi-modal 정보를 통한 더 강건한 pseudo-label 생성

**8.2 장거리 Domain Shift 대응**[11]

Synthetic-to-real이나 매우 다른 도메인 간 적응에서 CAT의 성능을 향상시키기 위해:

- Diffusion model을 통한 pseudo-target 생성과 함께 CAT 적용
- Open-set domain adaptation (타겟에 소스에 없는 클래스가 있는 경우)
- Partial domain adaptation (타겟이 소스의 부분집합만 포함)

**8.3 Scalability와 효율성**[13][9]

대규모 데이터셋과 모델에 대한 개선:

- $$L_c$$의 O(|X|²) 복잡도를 O(|X|log|X|) 또는 $$O(|X|)$$로 감소시키는 approximation
- 작은 메모리 footprint를 위한 mini-batch에서의 정확한 통계 추정
- 클러스터링 계산의 GPU 병렬화

**8.4 이상치(Outlier) 처리**[14]

현실의 데이터에서:

- 새로운 클래스 또는 배경 샘플의 robust 처리
- 결함이 있는 pseudo-label의 자동 감지 및 제거
- 점진적 신뢰도 업데이트 메커니즘

**8.5 Class Imbalance의 극복**[15][16]

CAT는 imbalanced 데이터에서 좋은 성능을 보이지만:

- 극단적 불균형(1000:1)에서의 강건성 검증
- Tail class의 prototypes를 명시적으로 강화하는 방법
- Long-tail recognition과의 통합

**8.6 Theoretical Analysis 심화**

CAT의 수렴성과 generalization bound를 엄밀하게 분석:

- Teacher-student 패러다임의 점근적 성능
- Pseudo-label 오류의 누적 효과 분석
- Clustering loss가 $$\epsilon_t(l_s, l_t)$$ 감소에 미치는 영향 정량화

***

### 9. 결론

**Cluster Alignment with a Teacher (CAT)**는 unsupervised domain adaptation의 역사에서 **중대한 전환점**입니다. 기존 방법들이 주변분포 정렬에만 집중한 반면, CAT는 **클래스-조건부 구조의 중요성**을 강조하고 이를 체계적으로 활용했습니다.

CAT의 핵심 혁신은 세 가지입니다:

1. **Discriminative Clustering Loss ($$L_c$$)**: 두 도메인에서 class-wise 응집력과 분리를 동시에 최적화
2. **Conditional Feature Matching Loss ($$L_a$$)**: 클래스별 중심을 정렬하여 class-conditional alignment 실현
3. **Robust Confidence-Thresholding**: 불확실한 샘플을 제외하여 안정성 증대

실증적으로는 **불균형 데이터셋에서의 극적 성능 향상**(27.4% → 98.8% in SVHN→MNIST)과 **기존 방법과의 호환성**이 CAT의 강점입니다.

최신 연구 동향(2024-2025)은 CAT의 정신을 이어받으면서 **Foundation Models 통합, Multi-source Adaptation, Source-free Scenarios**로 확장되고 있습니다. 특히 vision-language 모델의 등장으로 semantic alignment가 더욱 정교해지고 있으며, teacher-student 패러다임은 UDA의 표준 방식으로 정착되었습니다.

향후 연구는 다음과 같은 방향으로 진행될 것으로 예상됩니다:
- **효율성**: 대규모 모델과 데이터셋 대응
- **강건성**: Extreme domain shifts와 open-set scenarios
- **이론**: 수렴성과 generalization bound의 엄밀한 분석
- **실용성**: Real-world 도메인의 class imbalance, label noise, domain shift의 동시 처리

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/72c47543-7cd4-4d13-9bc0-bdf413940a67/1903.09980v2.pdf)
[2](http://arxiv.org/pdf/2103.13575.pdf)
[3](https://arxiv.org/pdf/1811.05443.pdf)
[4](https://arxiv.org/pdf/2209.15210.pdf)
[5](https://arxiv.org/html/2506.11493v1)
[6](https://www.ijcai.org/proceedings/2025/0356.pdf)
[7](https://www.sciencedirect.com/science/article/abs/pii/S0950705122013004)
[8](https://proceedings.nips.cc/paper/2021/file/c1fea270c48e8079d8ddf7d06d26ab52-Paper.pdf)
[9](http://arxiv.org/pdf/2302.06874.pdf)
[10](http://arxiv.org/pdf/2308.09931.pdf)
[11](https://arxiv.org/html/2510.03540v1)
[12](https://ghyeok.com/publication/2024_cvpr_a2xp/)
[13](http://arxiv.org/pdf/2403.14356.pdf)
[14](https://www.sciencedirect.com/science/article/abs/pii/S0167865524003209)
[15](https://pubmed.ncbi.nlm.nih.gov/38083470/)
[16](https://arxiv.org/abs/1910.10320)
[17](https://arxiv.org/pdf/2202.13310.pdf)
[18](http://arxiv.org/pdf/2410.02720.pdf)
[19](https://arxiv.org/pdf/2501.16410.pdf)
[20](https://arxiv.org/abs/2106.10812)
[21](https://arxiv.org/pdf/2212.02739.pdf)
[22](https://arxiv.org/html/2501.18592v1)
[23](https://cvpr.thecvf.com/virtual/2025/poster/32754)
[24](https://proceedings.mlr.press/v205/niemeijer23a/niemeijer23a.pdf)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC8134307/)
[26](https://www.sciencedirect.com/science/article/abs/pii/S0950705125005398)
[27](https://openaccess.thecvf.com/content/WACV2023/papers/Piva_Empirical_Generalization_Study_Unsupervised_Domain_Adaptation_vs._Domain_Generalization_Methods_WACV_2023_paper.pdf)
[28](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/01/ASRU2019_Domain_Adaptation_Encoder_Decoder.pdf)
[29](https://arxiv.org/html/2411.15557v4)
[30](https://arxiv.org/pdf/2311.08503.pdf)
[31](http://arxiv.org/pdf/2108.08995.pdf)
[32](https://arxiv.org/pdf/2212.07101.pdf)
[33](http://arxiv.org/pdf/2406.02024.pdf)
[34](https://arxiv.org/pdf/2110.09410.pdf)
[35](https://arxiv.org/html/2402.06809v2)
[36](https://www.sciencedirect.com/science/article/abs/pii/S0360544224012404)
[37](https://ieeexplore.ieee.org/document/10702166/)
