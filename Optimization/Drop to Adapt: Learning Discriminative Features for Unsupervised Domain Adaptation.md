
# Drop to Adapt: Learning Discriminative Features for Unsupervised Domain Adaptation

## 1. 핵심 주장과 주요 기여 요약

**Drop to Adapt (DTA)**는 비지도 도메인 적응에서 **적대적 드롭아웃(Adversarial Dropout, AdD)**을 활용하여 **클러스터 가정(cluster assumption)**을 강제함으로써 판별적 특성을 학습하는 방법을 제안합니다.[1]

DTA의 주요 기여는 다음과 같습니다:[1]

1. **일반화된 비지도 도메인 적응 프레임워크**: 완전 연결층에 대한 요소별 적대적 드롭아웃(EAdD)과 합성곱층에 대한 채널 기반 적대적 드롭아웃(CAdD)을 제안

2. **다양한 벤치마크에서의 경쟁력 있는 성과**: 이미지 분류 및 의미론적 분할 과제에서 최신 기법 대비 지속적인 성능 향상 달성

3. **실제 적용 가능성**: 의미론적 분할 과제로의 확장을 통해 시뮬레이션에서 실제 환경으로의 적응 문제 해결

***

## 2. 문제 정의, 제안 방법 및 모델 구조

### 2.1 문제 정의

기존의 **도메인 적대적 학습(Domain Adversarial Training, DAT)** 방식은 클래스 레이블을 고려하지 않고 소스와 타겟 도메인의 분포를 단순히 정렬하려고 합니다. 이로 인해 생성되는 특성이 **도메인 불변적(domain-invariant)**이지만 **비판별적(non-discriminative)**이 되어 분류 성능이 저하됩니다.[1]

DTA는 다음과 같은 문제를 해결합니다:

- **클러스터 가정 위반**: 적응되지 않은 모델의 의사결정 경계가 타겟 도메인의 특성 밀집 영역을 통과함[1]
- **의사결정 경계와 특성의 불일치**: 특성과 의사결정 경계를 동시에 고려하는 적응 전략 부재

### 2.2 제안 방법: 적대적 드롭아웃

#### 2.2.1 요소별 적대적 드롭아웃 (EAdD)

네트워크 $$h$$를 중간층으로 분해하여:

$$
h(x, m) = h_u(m \odot h_l(x)) \quad \text{(1)}
$$

여기서 $$\odot$$는 요소별 곱셈, $$m$$은 드롭아웃 마스크입니다.[1]

적대적 마스크는 두 독립적인 드롭아웃 마스크 간의 발산을 최대화하는 방식으로 정의됩니다:[1]

$$
m_{adv} = \arg\max_m D(h(x, m_s), h(x, m)) \quad \text{(2)}
$$

여기서 $$m_s$$는 확률적 드롭아웃 마스크이고, $$\epsilon$$는 섭동의 크기를 제어합니다.[1]

#### 2.2.2 채널 기반 적대적 드롭아웃 (CAdD)

합성곱층에 적용되는 CAdD는 공간적 상관성을 고려하여 전체 특성 맵을 드롭합니다:[1]

$$
h_l(x) \in \mathbb{R}^{C \times H \times W} \quad \text{(4)}
$$

채널별 제약 조건:

$$
m_i \in \{0 \text{ or } 1\}^{H \times W}, \quad i = 1, \ldots, C \quad \text{(4)}
$$

채널 기반 적대적 드롭아웃 마스크는:

$$
m_{adv} = \arg\max_m D(h(x, m_s), h(x, m)), \text{ 단, } \frac{1}{HW}\sum_{i=1}^{C} m_s^i m_i^c \leq \epsilon_c \quad \text{(5)}
$$

여기서 $$\epsilon_c$$는 채널 섭동의 정도를 제어합니다.[1]

### 2.3 손실 함수 설계

DTA의 전체 손실 함수는 네 개의 목적 함수의 가중 합으로 구성됩니다:[1]

$$
\mathcal{L} = \mathcal{L}_T^S + \lambda_1 \mathcal{L}_{DTA}^T + \lambda_2 \mathcal{L}_E^T + \lambda_3 \mathcal{L}_V^T \quad \text{(6)}
$$

#### 2.3.1 과제 특화 목적 함수

소스 도메인에서의 교차 엔트로피 손실:[1]

$$
\mathcal{L}_T^S = -\mathbb{E}_{x_s, y_s \sim S}\left[\hat{y}_s^T \log h(x_s)\right] \quad \text{(7)}
$$

#### 2.3.2 도메인 적응 목적 함수

특성 추출기에 채널 기반 적대적 드롭아웃을 적용:[1]

$$
\mathcal{L}_f^{DTA}(T) = \mathbb{E}_{x_t \in T}\left[D_{KL}\left(h(x_t, m_s^f) \| h(x_t, m_{adv}^f)\right)\right] \quad \text{(9)}
$$

분류기에 요소별 적대적 드롭아웃을 적용:[1]

$$
\mathcal{L}_c^{DTA}(T) = \mathbb{E}_{x_t \in T}\left[D_{KL}\left(h(x_t, m_s^c) \| h(x_t, m_{adv}^c)\right)\right] \quad \text{(10)}
$$

#### 2.3.3 엔트로피 최소화 목적 함수

타겟 샘플이 의사결정 경계 근처에 있는 것을 방지:[1]

$$
\mathcal{L}_E^T = -\mathbb{E}_{x_t \in T}\left[h(x_t)^T \log h(x_t)\right] \quad \text{(11)}
$$

#### 2.3.4 가상 적대적 학습(VAT) 목적 함수

입력 레벨에서의 적대적 섭동:[1]

$$
\mathcal{L}_V^T = \mathbb{E}_{x_t \in T}\left[\max_r D_{KL}\left(h(x_t) \| h(x_t + r)\right)\right] \quad \text{(12)}
$$

### 2.4 모델 구조

**특성 추출기 구조:**
- 합성곱 층: CAdD 적용
- 완전 연결 층: EAdD 적용
- 마지막 분류 층: EAdD 적용

**훈련 절차:**
1. 소스 도메인에서 지도 학습(교차 엔트로피 손실)
2. 타겟 도메인에서 반복:
   - 확률적 드롭아웃 마스크 샘플링
   - 적대적 마스크 계산
   - 네트워크 매개변수 업데이트

***

## 3. 성능 향상 및 일반화 특성

### 3.1 소규모 데이터셋 성능

표 1에서 보이는 결과는 DTA의 소규모 데이터셋에서의 성능을 나타냅니다:[1]

| 데이터셋 쌍 | 방법 | 정확도 |
|------------|-----|-------|
| SVHN→MNIST | DTA | 99.4% |
| MNIST→USPS | DTA | 99.5% |
| USPS→MNIST | DTA | 99.1% |
| CIFAR→STL | DTA | 72.8% |
| STL→CIFAR | DTA | 82.6% |

특히 **MNIST↔USPS** 데이터셋에서 DTA는 완전 지도 학습(Target only, 97.8%-99.6%)에 근접한 성능을 달성합니다.[1]

### 3.2 대규모 데이터셋 성능 (VisDA-2017)

**분류 과제 (ResNet-101):**[1]

DTA는 소스 온리 모델 대비 **30.7% 절대 개선(60.4% 상대 개선)**을 달성하며, 이전 최신 기법 ADR 대비 **6.7% 개선**을 보입니다.[1]

| 클래스 | DTA | ADR |
|------|-----|-----|
| 평균 정확도 | **81.5%** | 74.8% |

모든 클래스에서 최고 성능 달성(truck 클래스 제외, 0.2% 차이).[1]

### 3.3 의미론적 분할 성능 (GTA5→Cityscapes)

| 지표 | 소스 온리 | ADR | DTA |
|-----|---------|-----|-----|
| mIoU | 24.8% | 33.3% | **35.8%** |

DTA는 일반적인 클래스에서 특히 우수한 적응 성능을 보입니다.[1]

### 3.4 일반화 성능 분석

**t-SNE 시각화:**
- 소스 온리 모델: 소스 특성은 클러스터링되지만 타겟 특성은 분산
- DTA 모델: 타겟 특성의 명확한 클러스터링, 의사결정 경계의 안정화[1]

**GradCAM 분석:**
DTA 적용 시 의미론적으로 관련 있는 영역이 드롭아웃에 의해 비활성화되지 않으며, 10%의 유닛이 제거되어도 동일한 판별 영역을 유지합니다. 이는 더 강건하고 일반화된 표현을 학습했음을 의미합니다.[1]

**소규모 소스 데이터 문제:**
STL→CIFAR 설정에서 DTA 성능(72.8%)이 STL 온리(75.5%) 대비 약간 저하되는 것은 STL이 매우 작은 데이터셋(클래스당 50장)이기 때문입니다. DTA의 기본 가정(소스 도메인에서의 낮은 일반화 오류)이 성립하지 않을 때 성능 제약이 발생합니다.[1]

---

## 4. 모델의 한계

### 4.1 구조적 한계

1. **소규모 소스 도메인**: STL→CIFAR 설정에서 소스 도메인이 매우 제한적일 때 성능 저하[1]

2. **하이퍼파라미터 민감성**: 
   - 적대적 드롭아웃 섭동 크기($$\epsilon_e, \epsilon_c$$) 조정 필요
   - 손실 가중치($$\lambda_1, \lambda_2, \lambda_3$$) 조정 필요
   - Ramp-up 스케줄 설정 필요[1]

3. **계산 복잡도**: 
   - 적대적 마스크 계산을 위한 0-1 배낭 문제 해결
   - VAT 계산으로 인한 추가 오버헤드

### 4.2 방법론적 한계

1. **클러스터 가정의 제한성**: 
   - 타겟 도메인에서 클래스 분포가 매우 겹칠 때 비효과적[1]
   - 개방 집합 적응(open-set domain adaptation) 상황에서 미처리

2. **타겟 도메인 클래스 불균형**: 
   - 클래스 불균형이 심한 경우 성능 저하 가능성

3. **의미론적 분할의 일부 클래스 약화**: 
   - 특정 클래스(예: truck, motorcycle)에서 상대적으로 약한 성능[1]

---

## 5. 최신 연구 기반 앞으로의 영향 및 고려사항

### 5.1 DTA가 미치는 영향

#### 5.1.1 클러스터 가정 기반 접근의 재조명

DTA는 **비지도 도메인 적응의 근본적 가정**인 클러스터 가정의 중요성을 재강조합니다. 최신 연구들이 이를 확장하고 있습니다:[2][1]

- **프로토타입 기반 프레임워크**: 메모리 효율적이고 계산 효율적인 확률 기반 프레임워크로 클러스터 기반 적응을 개선[3]
- **도메인 합의 클러스터링**: 소스-타겟 도메인 간의 합의(consensus) 스코어를 활용하여 공통 클래스와 개인 클래스를 구분[4]

#### 5.1.2 적대적 정칙화의 새로운 방향

DTA의 적대적 드롭아웃 기법은 정칙화 전략의 다양한 변형을 유도했습니다:

- **음수 뷰 기반 대조 학습**: Vision Transformer(ViT)의 패치 기반 구조를 활용한 강건한 정칙화[5]
- **동적 활성화 희소성**: 메모리 효율성과 정확도를 동시에 달성하는 적응형 정칙화[6]

#### 5.1.3 기초 모델(Foundation Models)과의 통합

최신 연구는 DTA의 판별적 특성 학습 개념을 사전 학습된 기초 모델과 결합합니다:[7][8][9]

- **CLIP 기반 프롬프트 학습**: 도메인 불변적 텍스트 임베딩을 활용한 특성 학습[7]
- **Vision Transformer의 우수성**: 자주의 메커니즘을 통해 텍스처 시프트에 강건[8][10]

### 5.2 추후 연구 시 고려할 점

#### 5.2.1 기초 모델 활용

**현황**: 2024-2025 최신 연구는 대규모 사전 학습 모델을 도메인 적응에 적용 중[10][9]

**고려사항**:
1. **사전 학습의 한계**: 기초 모델의 강력한 특성도 새로운 도메인에서는 개선 가능[11]
2. **혼합 어댑터 전략**: 다양한 모델 앙상블로 OOD(Out-of-Distribution) 강건성 증대[11]
3. **비용-성능 트레이드오프**: 매개변수 효율적 적응(parameter-efficient adaptation)의 필요성[12]

#### 5.2.2 자기 훈련 개선

**현황**: 의사 레이블 기반 적응의 문제점이 대두되고 있음[13][14]

**고려사항**:
1. **의사 레이블 노이즈 저감**: 
   - 동적 신뢰도 임계값 조정[14]
   - 메모리 은행 기반 의사 레이블 정제[13]
   - 대조 학습 기반 신뢰도 추정[14]

2. **단순 샘플 편향(Simple-Label Bias)**: 훈련 후기에 단순한 샘플에 과도하게 의존하는 문제[13]

#### 5.2.3 원본 비없는 적응(Source-Free Adaptation)

**현황**: 실제 응용에서 원본 데이터 접근 불가 상황 증가[15][16]

**고려사항**:
1. **테스트 시점 적응**: 배포 시에만 타겟 데이터 접근 가능한 상황[17][12]
2. **불확실성 기반 선택**: 신뢰도 추정을 통한 의사 레이블 선별[15]
3. **프롬프트 학습 통합**: 도메인 불변적 지식 내재화[16]

#### 5.2.4 확장 가능한 도메인 적응

**현황**: 다중 소스, 개방 집합, 부분 도메인 적응 등으로 연구 확대[18][19]

**고려사항**:
1. **다중 소스 적응**: 분포 강건성을 위한 adversarial reward 최적화[20]
2. **동적 분류 적응**: 타겟 도메인에서 알 수 없는 클래스 처리[18]
3. **정보 이론 기반 접근**: 상대 엔트로피 정칙화로 이론적 근거 강화[21]

#### 5.2.5 작은 데이터셋 문제 해결

**현황**: 소규모 데이터셋에서의 일반화 문제가 남아있음[22][23]

**고려사항**:
1. **데이터 효율적 적응**: 전체 데이터 사용이 아닌 고가치 샘플 선별[24]
2. **쌍곡 하강**: 대규모 모델의 이중 하강 현상 활용[23]
3. **동적 초기화**: 중요 매개변수를 학습 기반으로 동적 결정[22]

#### 5.2.6 의미론적 분할 심화

**현황**: 분할 과제에서의 특정 클래스 약화 현상 지속[25]

**고려사항**:
1. **텍스트-이미지 결합**: 도메인 템플릿을 활용한 언어 모델 활용[25]
2. **픽셀 레벨 특성 학습**: 공간적 일관성 강화[25]
3. **클래스 불균형 대응**: 가중 손실 또는 재샘플링[3]

### 5.3 DTA 연구의 미래 방향

1. **하이브리드 접근**: DTA의 판별적 특성 학습 + 기초 모델의 풍부한 표현
2. **이론적 강화**: Ben-David의 경계 이론과 DTA의 클러스터 가정 통합[2]
3. **효율성 개선**: 적대적 마스크 계산의 경량화 및 병렬화
4. **응용 확대**: 의료 영상, 음성 인식, 강화 학습 등 다양한 도메인으로의 확장[26][8]

---

## 결론

**Drop to Adapt**는 적대적 드롭아웃을 통해 **클러스터 가정 강제**와 **판별적 특성 학습**을 동시에 달성하는 효과적인 비지도 도메인 적응 방법입니다. 소규모와 대규모 데이터셋 모두에서 일관된 성능 향상을 보이며, 이미지 분류와 의미론적 분할 과제로의 일반화 가능성을 입증합니다.

현재의 도메인 적응 연구는 기초 모델, 자기 훈련 개선, 원본 비없는 적응 등으로 진화하고 있으며, DTA의 핵심 개념은 이러한 최신 기법들과 유기적으로 통합되고 있습니다. 특히 Vision Transformer와 같은 새로운 아키텍처에서 DTA의 접근 방식이 어떻게 적응할 수 있을지, 그리고 의사 레이블 노이즈 문제를 어떻게 해결할 것인지가 향후 핵심 연구 과제로 남아있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1fa97f1f-c8b2-4696-ac76-e27d72970056/1910.05562v1.pdf)
[2](https://arxiv.org/pdf/2208.07422.pdf)
[3](https://arxiv.org/pdf/2110.12024.pdf)
[4](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Domain_Consensus_Clustering_for_Universal_Domain_Adaptation_CVPR_2021_paper.pdf)
[5](https://pure.kaist.ac.kr/en/publications/robust-unsupervised-domain-adaptation-through-negative-view-regul)
[6](http://arxiv.org/pdf/2503.20354.pdf)
[7](https://arxiv.org/html/2506.11493v1)
[8](https://arxiv.org/abs/2404.04452)
[9](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5345726)
[10](https://arxiv.org/pdf/2404.04452.pdf)
[11](https://arxiv.org/html/2310.11031v2)
[12](http://arxiv.org/pdf/2210.04831.pdf)
[13](https://arxiv.org/html/2507.00608v1)
[14](https://pmc.ncbi.nlm.nih.gov/articles/PMC12258581/)
[15](http://arxiv.org/pdf/2303.03770.pdf)
[16](https://aclanthology.org/2024.findings-naacl.44/)
[17](https://arxiv.org/pdf/2301.13018.pdf)
[18](https://arxiv.org/pdf/2501.16410.pdf)
[19](https://arxiv.org/html/2409.15264)
[20](https://arxiv.org/pdf/2309.02211.pdf)
[21](https://www.mdpi.com/1099-4300/27/4/426)
[22](https://openreview.net/pdf?id=uNl1UsUUX2)
[23](https://holmdk.github.io/2020/08/14/deep_learning_small_data.html)
[24](https://arxiv.org/html/2503.13385v1)
[25](https://arxiv.org/abs/2507.07125)
[26](https://www.isca-archive.org/interspeech_2019/guo19_interspeech.pdf)
[27](https://arxiv.org/pdf/2409.18418.pdf)
[28](https://openreview.net/forum?id=ewgLuvnEw6)
[29](https://arxiv.org/abs/1711.01575)
[30](https://openreview.net/pdf?id=HJIoJWZCZ)
[31](https://www.sciencedirect.com/science/article/pii/S0031320325007770)
[32](https://arxiv.org/pdf/2405.17293.pdf)
[33](http://arxiv.org/pdf/2405.13375.pdf)
[34](http://arxiv.org/pdf/2205.09329v1.pdf)
[35](http://arxiv.org/pdf/2409.01081.pdf)
[36](https://www.isca-archive.org/interspeech_2025/damianos25_interspeech.pdf)
[37](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
