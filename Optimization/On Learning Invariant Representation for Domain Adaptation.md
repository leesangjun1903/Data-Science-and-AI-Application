# On Learning Invariant Representation for Domain Adaptation

### 1. 핵심 주장과 주요 기여

이 논문은 도메인 적응(Domain Adaptation)의 기존 패러다임에 대한 **근본적인 재검토**를 제시합니다. 주요 주장은 다음과 같습니다.[1]

**핵심 주장:** 불변 표현(invariant representation) 학습과 소스 도메인의 작은 에러만으로는 타겟 도메인에서의 성공적인 일반화를 **보장할 수 없다**는 것을 이론적, 실험적으로 증명했습니다.[1]

**주요 기여:**

**첫째, 반례(Counterexample) 구성**: 완벽하게 정렬된 불변 표현을 학습하고 소스 에러를 최소화할수록 타겟 에러가 **증가**하는 1차원 예제를 제시했습니다. 변환 함수 $$g(x) = I_{x\leq0}(x) \cdot (x + 1) + I_{x>0}(x) \cdot (x - 1)$$를 통해 소스와 타겟 분포가 완벽히 정렬되지만, 모든 가설(hypothesis) $$h$$에 대해 $$\varepsilon_S(h \circ g) + \varepsilon_T(h \circ g) = 1$$이 됨을 보였습니다.[1]

**둘째, 일반화 상한(Generalization Upper Bound)**: 조건부 분포 이동(conditional shift)을 명시적으로 고려한 새로운 상한을 제시했습니다.[1]

$$\varepsilon_T(h) \leq \varepsilon_S(h) + d_{\tilde{H}}(D_S, D_T) + \min\{E_{D_S}[|f_S - f_T|], E_{D_T}[|f_S - f_T|]\}$$

여기서 세 항은 각각 소스 에러, 주변 분포 차이, **레이블 함수 간 거리**를 나타냅니다. 기존 Ben-David et al. 의 상한에 포함된 비관적인 $$\lambda^*$$ 항(최적 공동 에러)을 제거하고, 문제의 본질적 구조(조건부 이동)를 직접 반영합니다.[1]

**셋째, 정보 이론적 하한(Information-Theoretic Lower Bound)**: 불변 표현을 학습하는 모든 방법에 대한 공동 에러의 하한을 증명했습니다.[1]

$$\varepsilon_S(h \circ g) + \varepsilon_T(h \circ g) \geq \frac{1}{2}\left(d_{JS}(D_S^Y, D_T^Y) - d_{JS}(D_S^Z, D_T^Z)\right)^2$$

이는 소스와 타겟의 레이블 주변 분포($$D_S^Y, D_T^Y$$)가 크게 다를 때, 불변 표현 학습($$d_{JS}(D_S^Z, D_T^Z) = 0$$)이 오히려 타겟 에러를 증가시킨다는 **근본적 트레이드오프**를 특징짓습니다.[1]

### 2. 해결 문제, 제안 방법 및 모델 구조

#### 해결하고자 하는 문제

비지도 도메인 적응에서 레이블된 소스 도메인 $$\langle D_S, f_S \rangle$$과 레이블되지 않은 타겟 도메인 $$\langle D_T, f_T \rangle$$ 간의 분포 이동을 다룹니다. 특히 **조건부 이동(conditional shift)**이 존재할 때, 즉 클래스 조건부조건부 분포 $$P(X|Y)$$가 두 도메인 간에 변할 때 기존 방법들이 실패하는 이유를 규명합니다[1].

#### 제안 방법

논문은 알고리즘보다는 **이론적 통찰**을 제공합니다.

**상한 기반 충분 조건**: 성공적인 적응을 위해서는 다음 세 조건이 **동시에** 만족되어야 합니다:[1]

1. 작은 소스 에러: $$\tilde{\varepsilon}_S(h)$$ 최소화
2. 주변 분포 정렬: $$d_{\tilde{H}}(\tilde{D}_S, \tilde{D}_T)$$ 최소화  
3. **조건부 분포 근접성**: $$\min\{E_{D_S}[|f_S - f_T|], E_{D_T}[|f_S - f_T|]\}$$ 최소화

여기서 $$\tilde{H} := \{\text{sgn}(|h(x) - h'(x)| - t) | h, h' \in H, t \in [1]\}$$는 H의 대칭 차이 클래스입니다[1].

**데이터 의존적 상한(Theorem 4.2)**: Rademacher 복잡도를 활용한 실용적 버전도 제공합니다.[1]

$$\varepsilon_T(h) \leq \tilde{\varepsilon}_S(h) + d_{\tilde{H}}(\tilde{D}_S, \tilde{D}_T) + 2\text{Rad}_S(H) + 4\text{Rad}_S(\tilde{H}) + \min\{E_{D_S}[|f_S - f_T|], E_{D_T}[|f_S - f_T|]\} + O\left(\sqrt{\frac{\log(1/\delta)}{n}}\right)$$

**하한 기반 필요 조건(Lemma 4.8)**: Markov 체인 $$X \xrightarrow{g} Z \xrightarrow{h} \hat{Y}$$에서 Jensen-Shannon 거리를 이용합니다.[1]

$$d_{JS}(D_S^Y, D_T^Y) \leq d_{JS}(D_S^Z, D_T^Z) + \sqrt{\varepsilon_S(h \circ g)} + \sqrt{\varepsilon_T(h \circ g)}$$

Data Processing Inequality(Lemma 4.6)를 통해 $$d_{JS}(D_S^{\hat{Y}}, D_T^{\hat{Y}}) \leq d_{JS}(D_S^Z, D_T^Z)$$임을 보이고, AM-GM 부등식을 적용하여 Theorem 4.3를 도출합니다.[1]

#### 모델 구조

논문은 특정 모델을 제안하지 않지만, 분석 대상으로 **DANN(Domain Adversarial Neural Network)**[Ganin et al., 2016]을 사용합니다. DANN은 다음 구조를 가집니다:[1]

- **특징 추출기** $$g: X \rightarrow Z$$: 2개 합성곱 층 (5×5 커널, 10/20 채널)
- **분류기** $$h: Z \rightarrow Y$$: 완전 연결층 (1280 → 100 → 10)
- **도메인 판별기**: 500×100 완전 연결층 + 이진 출력[1]

### 3. 성능 향상 및 한계

#### 성능 측면

논문의 주요 목적은 **성능 개선이 아닌 이론적 한계 규명**입니다. 실험 결과는 이론을 **검증**하는 방향으로 설계되었습니다.[1]

**주요 실험 발견(MNIST/USPS/SVHN):**

- USPS→MNIST: 초기 정확도 55% → 최고점 후 하락 → 35% 수렴[1]
- SVHN→MNIST: 초기 빠른 상승 후 점진적 하락 추세[1]
- **공통 패턴**: 적응 초기 빠른 성능 향상 후, **과적합으로 인한 지속적 하락** (Least Square Fit의 음의 기울기)[1]

이는 레이블 분포가 다를 때($$D_S^Y \neq D_T^Y$$) 불변 표현 학습의 과도한 최적화가 역효과를 낸다는 Theorem 4.3의 예측과 일치합니다.[1]

#### 한계

**이론적 한계:**

1. **하한의 느슨함**: 반례에서 공동 에러가 1이지만, Theorem 4.3는 $$d_{JS}(D_S^Y, D_T^Y) = d_{JS}(D_S^Z, D_T^Z) = 0$$일 때 하한 0을 제공하여 trivial합니다.[1]

2. **연속 레이블 공간 미지원**: 회귀 문제는 다루지 않습니다(최근 COD가 해결).[2]

3. **알고리즘 부재**: 조건부 분포를 효과적으로 정렬하는 구체적 방법을 제시하지 않습니다.[1]

**실험적 한계:**

- 제한된 네트워크 용량 (2 Conv + 1 FC)로 현대 표준에 비해 단순합니다.[1]
- 단일 방법(DANN)만 검증하여 일반성 부족.[1]

### 4. 일반화 성능 향상 가능성

#### 핵심 통찰

논문은 일반화 향상을 위한 **설계 원칙**을 제시합니다:[1]

**원칙 1: 레이블 분포 정렬 필수성**  
Theorem 4.1의 세 번째 항 $$\min\{E_{D_S}[|f_S - f_T|], E_{D_T}[|f_S - f_T|]\}$$은 교차 도메인 에러($$\varepsilon_S(f_T)$$ 또는 $$\varepsilon_T(f_S)$$)와 동일합니다[1]. 이를 줄이려면:

- **타겟 레이블 분포 추정**: 위 연구들처럼 타겟 유사 레이블 재보정[3][4][5]
- **클래스별 조건부 정렬**: CCDA, μDAR처럼 클래스 조건부 MMD/CMMD 최소화[6][7]

**원칙 2: 불변성과 판별성의 균형**  
Lemma 4.8은 완전한 불변성($$d_{JS}(D_S^Z, D_T^Z) = 0$$)이 레이블 분포 차이만큼 에러를 증가시킴을 시사합니다. 해결책:[1]

- **조건부 불변 표현**: COD처럼 조건부 연산자 불일치 최소화[2]
- **인과 표현 학습**: ICRL처럼 독립 인과 관계 보존[8]

#### 최신 연구 영향 (2024-2025)

**조건부 정렬 방법론 발전:**

**CCA-LSC (2024)**: 레이블 이동 보정을 통한 대조적 조건부 정렬로 불균형 도메인 적응 해결. 본 논문의 조건부 분포 정렬 필요성을 직접 구현합니다.[5][3]

**IPCA (2025)**: 다중 소스 혼합 타겟 적응에서 가역 투영과 조건부 정렬을 결합하여 도메인별 정보 보존.[9]

**μDAR (2024)**: Temporal ensemble로 견고한 유사 레이블 생성 후, 클래스별 조건부 CMMD로 분포 정렬. 본 논문의 유사 레이블 신뢰성 문제를 완화합니다.[6]

**인과 추론 통합:**

**ICRL (2025)**: GAN 기반 정규 분포 정렬로 인과 요인 독립성 보장. 본 논문의 "조건부 분포 근접성" 조건을 인과 구조로 확장합니다.[8]

**CASUAL (2023-2024)**: 조건부 대칭 지지 발산 최소화로 레이블 이동 하 타겟 위험 상한 개선. Theorem 4.1의 직접적 후속 연구입니다.[4][10]

**확산 모델 활용:**

**NOCDDA (2025)**: 조건부 확산 모델로 고신뢰 유사 레이블 타겟 샘플 생성 및 조건부 분포 정렬. 본 논문의 $$\lambda^*$$ 문제를 생성 모델로 우회합니다.[11]

**Graph DA (2024)**: 그래프 구조 이동에 대해 조건부 구조 이동(CSS)과 레이블 이동(LS)을 동시 완화. 비유클리드 데이터로 확장합니다.[12]

### 5. 향후 연구에 미치는 영향 및 고려 사항

#### 패러다임 전환

본 논문은 도메인 적응 연구의 초점을 **"주변 분포 정렬"에서 "조건부 분포 정렬"로** 전환시켰습니다. 2024-2025년 최신 연구들이 이를 실증합니다.[13][3][4][6][2][1]

#### 향후 연구 방향

**이론적 측면:**

1. **더 강한 하한**: Theorem 4.3의 느슨함 개선 (예: Lin et al. 의 도메인 불변 특징 화이트닝 기법)[14]
2. **인과 발견 통합**: Wu et al. 의 인과/반인과 학습 구분처럼, 레이블 생성 메커니즘 변화 조건 명확화[15]
3. **연속 레이블 확장**: COD처럼 회귀 문제로 이론 일반화[2]

**방법론적 측면:**

1. **레이블 분포 추정**: GOLS처럼 온라인 레이블 이동 환경으로 확장[16]
2. **다중 소스 적응**: IPCA, MBDA 처럼 혼합 타겟 시나리오 대응[9]
3. **생성 모델 활용**: SDA의 합성 도메인 정렬처럼 확산 모델 기반 데이터 증강[17]

**실무적 고려 사항:**

**레이블 이동 진단 우선**: 적응 전 $$d_{JS}(D_S^Y, D_T^Y)$$ 추정으로 불변 표현 학습 적합성 판단.[1]

**조건부 정렬 기법 선택**: 
- 이미지 분류: CCDA의 클래스 조건부 판별기[7]
- 의료 영상: UDA-CDM의 조건부 확산 모델[18]
- 시계열: μDAR의 Temporal ensemble[6]

**과적합 모니터링**: Fig. 3의 2단계 패턴을 추적하여 조기 종료 적용.[1]

**하이브리드 접근**: LIRR처럼 불변 표현과 불변 위험을 **공동 학습**하여 준지도 설정 활용.[19]

#### 한계 인식

**가정의 현실성**: 본 논문의 이론은 타겟 레이블 함수 $$f_T$$ 접근을 가정하지만, 실무에서는 유사 레이블로 근사해야 합니다. CCA-LSC의 레이블 이동 보정 같은 추가 기법이 필요합니다.[5][1]

**계산 복잡도**: 조건부 정렬은 클래스별 통계 계산으로 복잡도가 증가합니다. 효율성 연구 필요.[7][6]

**도메인 특화 지식**: 인과 구조 발견은 종종 도메인 전문가 지식을 요구하여 확장성을 제한합니다.[20][8]

이 논문은 도메인 적응의 **근본적 한계**를 규명하여 맹목적 불변 표현 학습을 경계하게 했으며, 조건부 분포 정렬이라는 명확한 해결 방향을 제시하여 향후 5년 이상 연구 의제를 설정했습니다. 최신 방법들이 이를 다양한 형태로 구현하며 이론적 예측을 검증하고 있습니다.[3][4][13][11][5][8][6][2][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b2120ec7-e6f8-4460-b6b2-7b0175622f0d/1901.09453v2.pdf)
[2](https://arxiv.org/abs/2408.06638)
[3](https://link.springer.com/10.1007/978-3-031-78195-7_2)
[4](https://arxiv.org/html/2305.18458)
[5](https://arxiv.org/html/2412.20337v1)
[6](https://ieeexplore.ieee.org/document/10884097/)
[7](https://ieeexplore.ieee.org/document/10885014/)
[8](https://www.nature.com/articles/s41598-025-96357-0)
[9](https://ojs.aaai.org/index.php/AAAI/article/view/34111)
[10](https://arxiv.org/abs/2305.18458)
[11](https://www.ijcai.org/proceedings/2025/0193.pdf)
[12](https://arxiv.org/abs/2403.01092)
[13](http://proceedings.mlr.press/v97/zhao19a/zhao19a.pdf)
[14](https://www.sciencedirect.com/science/article/abs/pii/S0957417425033615)
[15](https://jmlr.org/papers/v25/22-1024.html)
[16](https://dl.acm.org/doi/10.1145/3690624.3709182)
[17](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Everything_to_the_Synthetic_Diffusion-driven_Test-time_Adaptation_via_Synthetic-Domain_Alignment_CVPR_2025_paper.pdf)
[18](https://ieeexplore.ieee.org/document/10635862/)
[19](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Learning_Invariant_Representations_and_Risks_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.html)
[20](https://aclanthology.org/2024.findings-emnlp.476.pdf)
[21](https://www.semanticscholar.org/paper/8d879d19f184c47eeea59b983725d0a1d7511227)
[22](https://journals.sagepub.com/doi/10.1177/09544062241274178)
[23](https://ieeexplore.ieee.org/document/10214107/)
[24](https://arxiv.org/html/2502.06272v1)
[25](https://arxiv.org/pdf/1811.05443.pdf)
[26](http://arxiv.org/pdf/2202.03628.pdf)
[27](http://arxiv.org/pdf/2103.13575.pdf)
[28](https://arxiv.org/html/2403.07798v1)
[29](https://arxiv.org/pdf/2310.04723.pdf)
[30](https://arxiv.org/html/2506.17137v2)
[31](https://www.sciencedirect.com/science/article/abs/pii/S0950705124009031)
[32](https://papers.neurips.cc/paper_files/paper/2022/file/8b21a7ea42cbcd1c29a7a88c444cce45-Paper-Conference.pdf)
[33](https://link.aps.org/doi/10.1103/PhysRevE.105.044306)
[34](https://francis-press.com/papers/19470)
[35](https://openreview.net/forum?id=ffBj12yh58)
[36](https://www.sciencedirect.com/science/article/abs/pii/S0031320325012889)
[37](https://arxiv.org/html/2412.10115v1)
[38](https://ink.library.smu.edu.sg/etd_coll/663/)
