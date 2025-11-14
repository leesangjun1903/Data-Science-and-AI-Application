# HoMM: Higher-order Moment Matching for Unsupervised Domain Adaptation

### 1. 핵심 주장 및 주요 기여

HoMM 논문은 **비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)에서 고차 모멘트 정합(Higher-order Moment Matching)**을 통한 도메인 간 분포 불일치 문제를 해결하는 방법을 제시합니다.[1]

**핵심 주장:**
- 기존 방법들은 1차(평균) 또는 2차(공분산) 통계량만 정합하지만, 심층 신경망의 특성 분포는 **비가우시안 분포(non-Gaussian distribution)**를 따릅니다.[1]
- 3차 및 4차 모멘트를 포함한 **고차 통계량 정합**이 더 정확한 분포 특성 표현을 가능하게 합니다.[1]

**주요 기여:**
1. **통합 프레임워크:** MMD(Maximum Mean Discrepancy)와 CORAL(Correlation Alignment)을 포함하는 통합 프레임워크로 임의 차수의 모멘트 텐서 정합 수행[1]
2. **수학적 동등성 증명:** 1차 HoMM = MMD, 2차 HoMM = CORAL 임을 증명[1]
3. **계산적 효율성:** 그룹 모멘트 정합(Group Moment Matching)과 무작위 샘플링 정합(Random Sampling Matching) 전략으로 $$O(L^p)$$의 지수적 복잡도를 $$O(ng \cdot \lfloor L/ng \rfloor^p)$$ 또는 $$O(N)$$으로 감축[1]
4. **의사 라벨 기반 판별적 클러스터링:** 신뢰도 높은 타겟 샘플에 의사 라벨을 할당하여 진정성 있는 표현 학습[1]

---

### 2. 해결하고자 하는 문제 및 제안 방법

#### 2.1 문제 정의

**도메인 이동(Domain Shift) 문제:**
- 소스 도메인에서 학습한 CNN 모델이 타겟 도메인에 배포될 때 성능이 급격히 감소합니다.[1]
- 타겟 도메인 데이터에는 라벨이 없으므로 지도학습 재학습이 불가능합니다.[1]

**기존 방법의 한계:**
MMD와 CORAL 같은 방법들은 다음 문제를 가집니다:
- $$L^1$$ 및 $$L^2$$ 통계량만 정합하므로 비가우시안 분포를 완전히 특성화할 수 없습니다.[1]
- 이로 인해 "coarse-grained" 도메인 정렬만 보장되며 세밀한 분포 정합이 불가능합니다.[1]

#### 2.2 제안 방법

**기본 고차 모멘트 정합 손실함수:**

$$
L_d = \frac{1}{L^p} \left\| \frac{1}{n_s} \sum_{i=1}^{n_s} \varphi_\theta(x_i^s)^{\otimes p} - \frac{1}{n_t} \sum_{i=1}^{n_t} \varphi_\theta(x_i^t)^{\otimes p} \right\|_F^2
$$

여기서 $$u^{\otimes p}$$는 $$p$$-레벨 텐서 거듭제곱으로, $$p$$차 고차 모멘트를 계산합니다.[1]

**1차 및 2차 특수화:**

- **1차 HoMM (p=1):**

$$
L_d = \frac{1}{L} \left\| \frac{1}{b} \sum_{i=1}^{b} h_i^s - \frac{1}{b} \sum_{i=1}^{b} h_i^t \right\|_F^2
$$

→ 선형 MMD와 동등[1]

- **2차 HoMM (p=2):**

$$
L_d = \frac{1}{L^2} \left\| \frac{1}{b} \sum_{i=1}^{b} h_i^{s^T} h_i^s - \frac{1}{b} \sum_{i=1}^{b} h_i^{t^T} h_i^t \right\|_F^2 = \frac{1}{b^2L^2} \|G(h^s) - G(h^t)\|_F^2
$$

→ 그램 행렬(Gram Matrix) 정합 = CORAL과 동등[1]

#### 2.3 계산적 최적화 기법

**그룹 모멘트 정합:**

$$
L_d = \frac{1}{b^2 \lfloor L/n_g \rfloor^p} \sum_{k=1}^{n_g} \left\| \sum_{i=1}^{b} h_i^{s,k \otimes p} - \sum_{i=1}^{b} h_i^{t,k \otimes p} \right\|_F^2
$$

숨겨진 뉴런을 $$n_g$$개 그룹으로 분할하여 복잡도를 $$O(n_g \cdot \lfloor L/n_g \rfloor^p)$$로 감축[1]

**무작위 샘플링 정합:**

$$
L_d = \frac{1}{b^2 N} \sum_{k=1}^{N} \left[ \sum_{i=1}^{b} \prod_{j=rnd[k,1]}^{rnd[k,p]} h_i^s(j) - \sum_{i=1}^{b} \prod_{j=rnd[k,1]}^{rnd[k,p]} h_i^t(j) \right]^2
$$

$$p$$-레벨 텐서에서 무작위로 $$N$$개 값을 선택하여 복잡도를 $$O(N)$$으로 감축[1]

#### 2.4 RKHS 확장

생식 커널 힐베르트 공간(Reproducing Kernel Hilbert Space, RKHS)으로 확장:

$$
L_d = \frac{1}{L^p} \left\| \frac{1}{b} \sum_{i=1}^{b} \psi(h_i^{s \otimes p}) - \frac{1}{b} \sum_{i=1}^{b} \psi(h_i^{t \otimes p}) \right\|_F^2
$$

커널화된 HoMM (KHoMM)은 RBF 커널을 사용하여 비선형 분포 정합을 수행합니다.[1]

#### 2.5 판별적 클러스터링

기존 엔트로피 최소화의 한계를 극복하기 위해 신뢰도 기반 의사 라벨링:

$$
L_{dc} = \frac{1}{n_t} \sum_{i=1}^{n_t} \| h_i^t - c_{\hat{y}_i^t} \|_2^2
$$

여기서 $$\hat{y}_i^t$$는 예측 확률이 임계값 $$\eta$$를 초과하는 샘플의 의사 라벨이고, $$c_j$$는 이동 평균(moving average)으로 업데이트되는 클래스 중심입니다.[1]

$$
c_j^{t+1} = \alpha c_j^t + (1-\alpha) \Delta c_j^t
$$

#### 2.6 전체 목적 함수

$$
L = L_s + \lambda_d L_d + \lambda_{dc} L_{dc}
$$

- $$L_s$$: 소스 도메인 분류 손실
- $$L_d$$: 고차 모멘트 정합 손실
- $$L_{dc}$$: 판별적 클러스터링 손실[1]

***

### 3. 모델 구조 및 구현

#### 3.1 네트워크 아키텍처

**두 스트림 CNN 구조:**
- 소스 도메인과 타겟 도메인 샘플을 처리하는 두 개의 동일한 스트림[1]
- 매개변수 공유(Tied Weights)를 통해 동일한 변환 학습[1]
- 마지막 완전 연결층(FC 레이어) 출력을 적응 레이어로 사용[1]

**중요 설계 결정:**
- 적응 레이어에는 **ReLU 활성화 함수를 사용할 수 없습니다** (대부분의 값이 0이 되어 고차 텐서 계산 불가)[1]
- 대신 **tanh 활성화 함수** 사용으로 모멘트 정보 보존[1]

#### 3.2 성능 향상 분석

**자릿수 인식 데이터셋(SVHN→MNIST):**
- CORAL: 89.5% → HoMM (p=3): 96.5% (+7.0%)[1]
- KHoMM (p=3): 97.2% (+7.7%)
- 의사 라벨 클러스터링 포함 전체 모델: 98.9%[1]

**Office-31 데이터셋:**
- CORAL: 79.3% (A→W) → HoMM (p=4): 89.8% (+10.5%)[1]
- 어려운 전이 작업 A→D에서 74.8% → 86.6% (+11.8%)[1]

**Office-Home 데이터셋 (더 어려운 벤치마크):**
- DAN: 57.0% (A→P) → HoMM (p=4): 63.5% (+6.5%)[1]
- JAN 대비 3-5% 성능 향상[1]

#### 3.3 모멘트 차수별 성능 분석

표 4 분석에서 나타난 현상:[1]
- $$p=1$$: 71.9% (SVHN→MNIST)
- $$p=2$$: 89.5% 
- $$p=3$$: 96.5% (최적)
- $$p=4$$: 95.7%
- $$p \geq 5$$: 성능 감소 (94.8%, 91.5% ...)

**해석:** 5차 이상 모멘트는 **샘플 부족 문제(Small Sample Size Problem)**로 인해 부정확하게 추정되어 일반화 성능 저하[2][1]

***

### 4. 일반화 성능 향상 메커니즘

#### 4.1 분포 특성화의 개선

**비가우시안 분포 대응:**
- 심층 신경망 특성은 복잡한 다봉우리 분포를 나타냅니다.[1]
- 2차 통계량(공분산)만으로는 이러한 구조를 충분히 표현 불가합니다.[1]
- 3차 모멘트(왜도, Skewness)와 4차 모멘트(첨도, Kurtosis)는 분포의 비대칭성과 꼬리 특성을 캡처합니다.[1]

**t-SNE 시각화:**
Figure 4에서 확인 가능한 개선 사항:
- Source Only: 도메인 간 큰 분포 차이[1]
- KMMD/CORAL: 전역 분포 정렬이나 클래스 간 경계가 불명확[1]
- HoMM (p=3): 클래스 간 분리가 명확하고 도메인 간 정렬이 우수[1]
- Full Loss (클러스터링 포함): 가장 명확한 클래스 분리와 도메인 정렬[1]

#### 4.2 판별성 보존

**기존 엔트로피 최소화의 문제점:**
- 초기 단계에서 신뢰도 낮은 샘플도 과신(overconfidence) 유도[1]
- 잘못 분류된 샘플도 엔트로피 감소 압력에 의해 강하게 분류[1]

**제안된 해결책 (의사 라벨 기반 클러스터링):**
- 예측 확률 > η (임계값)인 샘플만 선택[1]
- 이들을 공유 특성 공간에서 클래스 중심으로 당김[1]
- 파라미터 민감도 분석에서 최적 η = 0.75-0.85[1]

#### 4.3 다중 도메인 정렬의 효과

**수학적 근거:**
도메인 적응 일반화 오차 상한에 대한 이론적 분석:

모멘트 기반 도메인 적응의 이론에 따르면, 타겟 오류의 상한은:[3]

$$
\epsilon_T(\hat{h}) \leq \epsilon_T(h_T^*) + d_{CM_k}(D_S, D_T) + O(\text{complexity terms})
$$

여기서 $$d_{CM_k}$$는 k차 교차 모멘트 발산입니다.[3]

고차 모멘트 정합은 이 상한을 직접적으로 최소화하므로 더 강한 이론적 보장을 제공합니다.[1]

***

### 5. 논문의 한계

#### 5.1 활성화 함수 제약

- ReLU 불가 → tanh 필수 (대부분 심층 신경망과 불호환)[1]
- 이는 ImageNet 사전학습 모델 활용 시 추가 미세조정 필요[1]

#### 5.2 하이퍼파라미터 민감성

- $$\lambda_d$$ (도메인 정합 가중치): 1부터 $$10^8$$까지 광범위한 범위[1]
- 4차 모멘트가 3차보다 훨씬 큰 $$\lambda_d$$ 필요 (자릿수 데이터셋에서 $$10^4$$ vs $$10^7$$)[1]
- 이는 깊은 특성 값이 작을 때 고차 모멘트가 거의 0에 가까워져 스케일 조정 필요[1]

#### 5.3 표본 부족 문제

- 5차 이상 모멘트는 정확도 급감[1]
- 배치 크기 제약으로 인한 추정 불안정성[1]

#### 5.4 계산 비용

- 그룹 모멘트 정합: 여전히 $$n_g$$ 그룹에 대해 순차 처리[1]
- 무작위 샘플링: $$N=1000$$으로 설정 시 고정 비용이지만 정보 손실 가능[1]

***

### 6. 최신 연구 동향 및 향후 고려사항

#### 6.1 최신 연구 트렌드 (2023-2025)

**1. 사전학습 모델 기반 접근**
CLIP(Contrastive Language-Image Pretraining) 기반 방법들이 새로운 표준으로 부상:[4]
- 다중 모드 의미 정보를 통한 강력한 제로샷 능력[4]
- 기존 모멘트 정합보다 우수한 도메인 불변성[4]
- 2024-2025년 연구에서 급속히 성장하는 분야[4]

**2. 정보 이론 기반 접근**
- 상대 엔트로피 정규화 통합:[5]
- 정보-이론적 상한을 통한 이론적 근거 강화[5]
- 분산 강건 학습(Distributionally Robust Learning)의 발전[6]

**3. 위험 분포 정합 (Risk Distribution Matching)**
- 스칼라 위험 분포 정합으로 고차원 문제 회피:[7]
- 계산 효율성 향상 및 일반화 성능 개선[7]
- 최악의 케이스 도메인과 전체 도메인 분포 정합 전략[7]

**4. 컨텍스트 인식 도메인 적응**
- 배치 학습의 한계 극복:[8]
- 전역 통계 및 기하학적 특성 통합:[8]
- 미니배치 기반 불안정성 감소[8]

#### 6.2 HoMM 논문이 최신 연구에 미친 영향

**직접적 영향:**
- 282회 인용 (Google Scholar)[9]
- HOMDA (High-Order Moment-Based Domain Alignment, 2023)가 HoMM을 개선한 부분 최적 수송(POT) 전략으로 확장:[10]
- 텐서 기반 도메인 정합의 이론적 토대 제공[1]

**간접적 영향:**
- 고차 통계량의 중요성 강조로 다양한 응용 분야 확대[11][12]
- 의사 라벨 기반 클러스터링 전략이 현대 semi-supervised learning 방법에 통합[2]

#### 6.3 향후 연구 시 고려할 점

**1. 활성화 함수 문제 해결**
- 현대 신경망과 호환 가능한 고차 모멘트 계산 방법 개발
- ReLU의 특수성을 활용한 모멘트 근사 (예: Gumbel-Softmax 대안)
- 비음(non-negative) 특성 공간에서의 모멘트 정합 이론화

**2. 표본 효율성 개선**
- 소 배치에서의 모멘트 추정 안정화 알고리즘
- 적응형 모멘트 차수 선택 메커니즘
- 불확실성 추정을 통한 신뢰도 가중 정합

**3. 이론-실무 간 격차 해소**
- 비가우시안 분포에 대한 더 강한 일반화 상한
- 클러스터링 손실과 모멘트 정합의 상호작용 이론화
- 임계값 기반 의사 라벨링의 수렴성 보증

**4. 확장성 개선**
- 초고차 도메인(>4차)에 대한 효율적 계산 방법
- 다중 소스 도메인 적응으로의 확장 (M3SDA 스타일)
- 지속적 도메인 시프트 학습(Continual Domain Adaptation) 통합[11]

**5. 멀티모달 및 사전학습 통합**
- CLIP/DINO 같은 기초 모델의 특성 공간에서의 모멘트 정합
- 텍스트-이미지 정렬을 활용한 도메인 불변 특성 학습[13]
- 프롬프트 학습과 고차 모멘트 정합의 통합[4]

**6. 응용 확대**
- 의료 이미징 도메인 적응 (분포 이동 심각)[14]
- 그래프 기반 데이터의 고차 모멘트 정합[15]
- 자율주행 및 로봇 공학의 시뮬레이션-실제 이전[12]

***

### 결론

HoMM 논문은 **단순하지만 강력한 통찰**을 제시합니다: 도메인 간 분포 정합 시 고차 통계량을 활용하면 비가우시안 특성 분포를 더 정확히 특성화할 수 있다는 점입니다. 이는 다음과 같은 영향을 미쳤습니다:[1]

1. **방법론적 기여:** MMD와 CORAL의 일반화 프레임워크 제시
2. **실용적 성과:** 여러 벤치마크에서 10% 이상의 성능 향상
3. **이론적 동기:** 고차 통계량의 중요성에 대한 수학적 근거 제공

그러나 현대의 **자기지도학습(Self-Supervised Learning)**, **사전학습 모델 활용**, **정보 이론 기반 접근**의 성장으로 인해 향후 연구는 다음 방향으로 진화할 것으로 예상됩니다: (1) HoMM의 원리를 보존하면서 계산 효율성을 극대화하는 적응형 방법, (2) 기초 모델(Foundation Models)과의 통합, (3) 이론-실무 간 격차를 좁히는 강화된 일반화 보증.

***

### 참고 문헌 (본 보고서의 정보 출처)

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3c94ef3f-382c-47ee-8aec-652ab4d3ad2d/1912.11976v1.pdf)
[2](https://openreview.net/pdf?id=3qMwV98zLIk)
[3](http://arxiv.org/pdf/2002.08260.pdf)
[4](https://arxiv.org/abs/2504.14280)
[5](https://www.mdpi.com/1099-4300/27/4/426)
[6](https://arxiv.org/pdf/2309.02211.pdf)
[7](https://openaccess.thecvf.com/content/WACV2024/papers/Nguyen_Domain_Generalisation_via_Risk_Distribution_Matching_WACV_2024_paper.pdf)
[8](https://arxiv.org/html/2502.06272v1)
[9](https://arxiv.org/abs/1912.11976)
[10](https://www.sciencedirect.com/science/article/abs/pii/S0950705122013016)
[11](https://arxiv.org/pdf/2303.15833.pdf)
[12](https://arxiv.org/html/2410.15811v2)
[13](https://cvpr.thecvf.com/virtual/2025/poster/34783)
[14](https://pmc.ncbi.nlm.nih.gov/articles/PMC9011180/)
[15](https://openaccess.thecvf.com/content/CVPR2024/html/Piao_Improving_Out-of-Distribution_Generalization_in_Graphs_via_Hierarchical_Semantic_Environments_CVPR_2024_paper.html)
[16](https://arxiv.org/pdf/2208.07422.pdf)
[17](https://arxiv.org/pdf/2110.12024.pdf)
[18](http://arxiv.org/pdf/2303.03770.pdf)
[19](https://openreview.net/forum?id=ewgLuvnEw6)
[20](https://openreview.net/forum?id=jeNWwtIX71)
[21](https://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.pdf)
[22](https://arxiv.org/abs/2507.07125)
[23](https://arxiv.org/abs/2307.12622)
[24](https://www.sciencedirect.com/science/article/pii/S0924271625001224)
[25](https://arxiv.org/pdf/2101.09979.pdf)
[26](http://arxiv.org/pdf/2406.14828.pdf)
[27](https://arxiv.org/pdf/2106.11344.pdf)
[28](http://arxiv.org/pdf/0902.3430.pdf)
[29](https://arxiv.org/pdf/1502.02791.pdf)
[30](https://divamgupta.com/unsupervised-learning/2020/10/31/pseudo-semi-supervised-learning-for-unsupervised-clustering.html)
[31](https://www.jmlr.org/papers/volume26/24-0737/24-0737.pdf)
[32](https://aclanthology.org/2024.findings-acl.725/)
[33](https://arxiv.org/abs/2411.07957)
[34](https://aclanthology.org/2024.acl-long.715.pdf)
[35](https://www.sciencedirect.com/science/article/pii/S0957417423010904)
[36](https://www.sciencedirect.com/science/article/abs/pii/S0167865523002556)
[37](https://arxiv.org/pdf/2108.13624.pdf)
[38](https://arxiv.org/pdf/2411.02444.pdf)
[39](http://arxiv.org/pdf/2406.02024.pdf)
[40](http://arxiv.org/pdf/2403.05523.pdf)
[41](http://arxiv.org/pdf/2106.04496.pdf)
[42](http://arxiv.org/pdf/2103.02503v1)%3C%22.pdf)
[43](https://arxiv.org/pdf/2103.03097.pdf)
[44](https://arxiv.org/pdf/2404.04669.pdf)
[45](https://www.nature.com/articles/s43246-024-00731-w)
[46](https://arxiv.org/abs/2108.11974)
[47](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Aligning_Infinite-Dimensional_Covariance_CVPR_2018_paper.html)
[48](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d3b57e06e3fc45f077eb5c9f28156d4-Paper-Conference.pdf)
[49](https://teazrq.github.io/SMLR/reproducing-kernel-hilbert-space.html)
[50](https://openreview.net/forum?id=VXak3CZZGC)
[51](https://pmc.ncbi.nlm.nih.gov/articles/PMC10585553/)
[52](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)
[53](https://arxiv.org/pdf/1702.08811.pdf)
[54](https://www.matec-conferences.org/articles/matecconf/pdf/2020/15/matecconf_acmme20_03001.pdf)
[55](http://arxiv.org/abs/2502.00052)
[56](https://arxiv.org/pdf/2004.10618.pdf)
[57](http://arxiv.org/pdf/1704.04235.pdf)
[58](http://arxiv.org/pdf/2412.16255.pdf)
[59](https://arxiv.org/abs/1702.08811)
[60](https://www.ijcai.org/proceedings/2025/0710.pdf)
[61](https://www.mecs-press.org/ijieeb/ijieeb-v16-n2/IJIEEB-V16-N2-4.pdf)
[62](https://openreview.net/pdf?id=SkB-_mcel)
[63](https://proceedings.mlr.press/v238/gong24b/gong24b.pdf)
[64](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tang_Unsupervised_Domain_Adaptation_via_Structurally_Regularized_Deep_Clustering_CVPR_2020_paper.pdf)
[65](https://github.com/wzell/cmd)
[66](https://www.math.tsukuba.ac.jp/~aoshima-lab/abst_2024/15.Yao.pdf)
[67](https://arxiv.org/html/2301.11405v3)
[68](https://www.semanticscholar.org/paper/Central-Moment-Discrepancy-(CMD)-for-Representation-Zellinger-Grubinger/01dc0a157e355ddc34a426f121fc871601fda567)
