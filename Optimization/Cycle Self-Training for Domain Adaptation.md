# Cycle Self-Training for Domain Adaptation

## 1. 핵심 주장과 주요 기여

Cycle Self-Training (CST)은 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)에서 발생하는 **분포 변화로 인한 의사 레이블의 불안정성 문제**를 해결하기 위해 제안된 방법론입니다. 기존의 표준 자가 학습(self-training) 방법이 도메인 변화 상황에서 의사 레이블의 품질이 급격히 저하되는 한계를 극복하기 위해, CST는 순환 메커니즘을 통해 의사 레이블이 도메인 간 일반화되도록 명시적으로 강제합니다.[1]

**주요 기여는 다음과 같습니다:**

**Forward Step (순방향 단계)**: 소스 도메인에서 학습된 분류기 $$\theta_s$$를 사용하여 타겟 도메인에 대한 의사 레이블을 생성합니다:[1]

$$
y' = \arg\max_i \{f_{\theta_s,\phi}(x)[i]\}
$$

**Reverse Step (역방향 단계)**: 생성된 의사 레이블로 타겟 분류기 $$\hat{\theta}_t(\phi)$$를 학습하고, 이 분류기가 소스 도메인에서도 잘 작동하도록 공유 특징 추출기 $$\phi$$를 업데이트합니다:[1]

$$
\hat{\theta}_t(\phi) = \arg\min_\theta \mathbb{E}_{x\sim\hat{Q}}\ell(f_{\theta,\phi}(x), y')
$$

$$
\min_{\theta_s,\phi} \mathcal{L}_{Cycle}(\theta_s, \phi) := \mathcal{L}_{\hat{P}}(\theta_s, \phi) + \mathcal{L}_{\hat{P}}(\hat{\theta}_t(\phi), \phi)
$$

**Tsallis 엔트로피 정규화**: 표준 Gibbs 엔트로피가 과신뢰(over-confidence)를 유발하는 문제를 해결하기 위해, Tsallis 엔트로피를 도입했습니다:[1]

$$
S_\alpha(y) = \frac{1}{\alpha-1}\left(1 - \sum_i y_i^\alpha\right)
$$

여기서 엔트로피 지수 $$\alpha$$는 자동으로 최적화되어, 불확실성 측정의 유연성을 제공합니다.[1]

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 문제 정의

**분포 변화 하에서 의사 레이블의 불신뢰성**: 연구진은 VisDA-2017 데이터셋에서 실험적 분석을 수행한 결과, 소스와 타겟이 동일한 분포일 때는 의사 레이블 분포가 실제 레이블과 거의 일치하지만, covariate shift나 label shift가 존재할 경우 의사 레이블이 특정 클래스에 편향되고, 여러 클래스의 샘플이 다른 클래스로 대부분 잘못 분류됨을 확인했습니다.[1]

총 변동(Total Variation, TV) 거리 분석 결과, 표준 자가 학습에서 $$d_{TV}$$가 훈련 내내 약 0.26으로 유지되어, 의사 레이블 정확도가 최대 74%로 제한됨을 밝혔습니다.[1]

**기존 의사 레이블 선택 전략의 한계**: 엔트로피나 신뢰도 기반 임계값 설정 방법은 도메인 변화가 없을 때는 효과적(AUC=0.89)이지만, 도메인 변화가 있을 때 성능이 급격히 하락(AUC=0.78)합니다.[1]

### 2.2 제안된 방법론

**이중 수준 최적화 (Bi-level Optimization)**: CST는 이중 수준 최적화 문제로 정식화됩니다:[1]
- **내부 루프**: 의사 레이블로 타겟 분류기 학습
- **외부 루프**: 타겟 분류기가 소스 도메인에서 잘 작동하도록 특징 추출기 업데이트

이를 효율적으로 구현하기 위해, 경량 선형 헤드 $$\theta_t$$의 해석적 해를 계산하고 MAML처럼 2차 미분을 계산하지 않고 직접 역전파합니다.[2][1]

**Tsallis 엔트로피를 통한 적응적 불확실성 제어**: 최적 $$\alpha$$ 값은 다음을 최소화하여 자동으로 찾습니다:[1]

$$
\hat{\alpha} = \arg\min_{\alpha \in [1,2]} \mathcal{L}_{\hat{P}}(\hat{\theta}_{t,\alpha}, \phi)
$$

$$\alpha$$가 1에 가까우면 불확실한 예측에 대한 패널티가 크고, $$\alpha$$가 2에 가까우면 여러 점수가 유사한 것을 허용합니다. 이는 잘못된 의사 레이블이 1에 가까운 값으로 고정되어 수정될 수 없는 Gibbs 엔트로피의 문제를 해결합니다.[1]

## 3. 모델 구조

CST의 전체 구조는 다음 구성 요소로 이루어집니다:[1]

**특징 추출기 (Feature Extractor)** $$h_\phi$$: ResNet-50 (비전 태스크) 또는 BERT (언어 태스크)
**소스 분류기** $$g_{\theta_s}$$: 소스 도메인 학습용 선형 헤드
**타겟 분류기** $$g_{\hat{\theta}_t(\phi)}$$: 의사 레이블로 학습되는 타겟 헤드

**전체 학습 목적 함수**:[1]

$$
\min_{\theta_s,\phi} \mathcal{L}_{\hat{P}}(\theta_s, \phi) + \mathcal{L}_{\hat{P}}(\hat{\theta}_t(\phi), \phi) + \mathcal{L}_{\hat{Q},Tsallis,\hat{\alpha}}(\theta_s, \phi)
$$

이 구조는 표준 자가 학습과 달리 두 개의 분류기를 동시에 학습하며, 의사 레이블의 도메인 간 일반화를 명시적으로 강제합니다.

## 4. 성능 향상 및 실험 결과

### 4.1 주요 벤치마크 성능

**Office-Home (ResNet-50)**: CST는 12개 태스크 중 9개에서 최고 성능을 달성했으며, 평균 정확도 73.0%로 이전 SOTA인 SENTRY(72.2%)를 능가했습니다.[1]

**VisDA-2017**: 
- ResNet-50: CST 79.9% ± 0.5%, CST+SAM 80.6% ± 0.5%
- ResNet-101: CST 84.8% ± 0.6%, CST+SAM 86.5% ± 0.7%
- 이전 SOTA(STAR 82.7%)를 크게 상회했습니다.[1]

**Amazon Review (BERT 기반 감성 분류)**: 12개 태스크 모두에서 평균 정확도 91.5%로, 이전 방법들(VAT 90.1%, MDD 90.0%)을 상당히 능가했습니다.[1]

### 4.2 Ablation Study 결과

VisDA-2017에서의 ablation study는 각 구성 요소의 중요성을 보여줍니다:[1]

| 방법 | 정확도 | $$d_{TV}$$ |
|------|--------|------------|
| FixMatch | 74.5% ± 0.2% | 0.22 |
| FixMatch+Tsallis | 76.3% ± 0.8% | 0.15 |
| CST w/o Tsallis | 72.0% ± 0.4% | 0.16 |
| CST+Entropy | 76.2% ± 0.6% | 0.20 |
| **CST** | **79.9% ± 0.5%** | **0.12** |

- Tsallis 엔트로피가 표준 엔트로피보다 3.7% 더 효과적
- 순환 메커니즘만으로도 FixMatch 대비 5.4% 성능 향상
- CST는 의사 레이블과 실제 레이블 간 TV 거리를 0.12로 감소시켜 표준 자가 학습(0.22)보다 훨씬 더 신뢰할 수 있는 의사 레이블 생성[1]

### 4.3 의사 레이블 품질 분석

훈련 중 의사 레이블 오류 추적 결과, CST는 타겟 분류기 $$\theta_t$$의 소스 도메인 오류가 빠르게 감소하면서 동시에 의사 레이블 오류와 TV 거리가 지속적으로 감소함을 확인했습니다. 이는 표준 자가 학습에서 $$d_{TV}$$가 훈련 내내 거의 변하지 않는 것과 대조적입니다.[1]

## 5. 이론적 분석 및 한계

### 5.1 이론적 보장

**Theorem 1 (Expansion 가정 하에서 CST의 성능)**: $$(q, \epsilon)$$-constant expansion 가정 하에, 소스 모델 $$f_s$$와 타겟 모델 $$f_t$$가 타겟 도메인에서 유사하게 동작하고 $$f_t$$가 입력 변화에 강건할 때, 다음이 성립합니다:[1]

$$
\text{Err}_Q(f_s) \leq \text{Err}_P(f_t) + c + 2q + \frac{\rho}{\min\{\epsilon, q\}}
$$

**Theorem 2 (유한 샘플 보장)**: CST 목적 함수의 최소화자는 타겟 도메인에서 낮은 일반화 오류를 보장합니다:[1]

$$
\text{Err}_Q(f_s) \leq \mathcal{L}_{CST}(f_s, f_t) + 2q + \frac{4K}{\gamma}\left(\hat{\mathcal{R}}(\mathcal{F}|\hat{P}) + \hat{\mathcal{R}}(\tilde{\mathcal{F}} \times \mathcal{F}|\hat{Q})\right) + \frac{2}{\tau}\left(\hat{\mathcal{R}}(\mathcal{F}|\hat{P}) + \hat{\mathcal{R}}(\mathcal{F}|\hat{Q})\right) + \zeta
$$

### 5.2 Hard Case 분석

이차 신경망 $$f_{\theta,\phi}(x) = \theta^\top(\phi^\top x)^{\odot 2}$$에서, 소스 도메인에는 여러 해가 존재하지만 타겟에서 작동하는 해는 하나뿐인 경우:

**Theorem 3 (Feature Adaptation과 Standard Self-Training의 실패)**: 높은 확률로 feature adaptation과 표준 자가 학습은 타겟 도메인에서 $$\epsilon$$ 이상의 오류를 가집니다.[1]

**Theorem 4 (CST의 성공)**: 높은 확률로 CST는 타겟 실제 레이블을 완벽히 복구합니다($$\text{Err}_Q = 0$$).[1]

### 5.3 방법론의 한계

**계산 복잡도**: 이중 수준 최적화와 $$\alpha$$ 값의 반복적 탐색으로 인해 표준 자가 학습보다 계산 비용이 증가합니다. 그러나 연구진은 해석적 해 계산과 이산화된 $$\alpha$$ 탐색으로 이를 완화했습니다.[1]

**Expansion 가정의 제약**: 이론적 보장은 $$(q, \epsilon)$$-constant expansion 가정에 의존하는데, 이는 조건부 분포 $$P_i$$와 $$Q_i$$가 가까이 위치하고 규칙적인 형태를 가진다는 것을 의미합니다. 실제 데이터에서 이 가정이 항상 성립하지는 않습니다.[1]

**의사 레이블 노이즈 완화의 한계**: Tsallis 엔트로피와 순환 메커니즘이 의사 레이블 품질을 크게 개선하지만, 도메인 간 격차가 매우 큰 경우 여전히 노이즈가 존재할 수 있습니다.[1]

## 6. 모델의 일반화 성능 향상

### 6.1 일반화 메커니즘

CST의 일반화 성능 향상은 다음 메커니즘을 통해 달성됩니다:

**도메인 간 일반화 강제**: 역방향 단계에서 타겟 분류기가 소스 도메인에서도 잘 작동하도록 강제함으로써, 의사 레이블이 도메인에 무관한 특징을 학습하도록 유도합니다.[1]

**적응적 불확실성 제어**: Tsallis 엔트로피의 $$\alpha$$ 매개변수를 자동으로 조정하여, 과도하게 확신하는 잘못된 예측을 방지하고 의사 레이블 수정 가능성을 유지합니다.[1]

**강건성(Robustness)과 불확실성의 연결**: Theorem 2에서 불확실성 $$1 - \mathbb{E}M(f_t(x))$$가 강건성 $$R(f_t)$$와 밀접하게 연관됨을 보여, 불확실성 최소화가 도메인 간 일반화로 이어짐을 이론적으로 정당화했습니다.[1]

### 6.2 실험적 증거

**다양한 도메인 갭에서의 안정성**: Office-Home, VisDA-2017, Amazon Review 등 다양한 벤치마크에서 일관된 성능 향상을 보여, 방법론의 일반성을 입증했습니다.[1]

**아키텍처 독립성**: ResNet-50, ResNet-101, BERT 등 다양한 백본에서 효과적으로 작동하여, 특정 모델에 종속되지 않는 일반화 능력을 보여줍니다.[1]

**CST+SAM 조합**: Sharpness-Aware Minimization (SAM)과 결합하여 함수 클래스의 Lipschitz 연속성을 정규화함으로써 추가적인 일반화 성능 향상을 달성했습니다(VisDA-2017에서 86.5%).[1]

## 7. 미래 연구에 미치는 영향 및 고려 사항

### 7.1 최신 연구 동향 (2023-2025)

**Meta-Tsallis Entropy 확장**: Lu et al. (2023)은 CST의 Tsallis 엔트로피 개념을 메타학습과 결합하여 인스턴스별 적응적 $$\alpha$$ 값을 학습하는 MTEM을 제안했습니다. BERT 기반 텍스트 분류에서 평균 4% 성능 향상을 달성했습니다.[3][4]

**Test-Time Adaptation (TTA)로의 확장**: 
- **Tent (2024)**: 엔트로피 최소화를 통한 완전 test-time adaptation을 제안했으나, 모델 붕괴 위험이 있습니다.[5][6]
- **Ranked Entropy Minimization (2025)**: 엔트로피 순위를 보존하면서 점진적 마스킹 전략을 통해 모델 붕괴 문제를 완화했습니다.[7][8]
- **Protected Test-Time Adaptation (2024)**: 베팅 마틴게일을 이용한 온라인 통계적 프레임워크로 분포 변화를 감지하고 엔트로피 매칭을 수행합니다.[2]

**Source-Free Domain Adaptation (SFDA) 발전**: 
- **UP2D (2025)**: 불확실성 인식 점진적 의사 레이블 디노이징을 통해 의료 영상 분할에서 SFDA 성능을 개선했습니다.[9]
- **Metric Learning 기반 SFDA (2024)**: 메트릭 학습을 통해 의사 레이블의 신뢰도를 향상시켰습니다.[10]

**Continual Test-Time Adaptation (CTTA)**: 
- **ViDA (2024)**: 동적 데이터 분포 하에서 오류 축적과 치명적 망각을 방지하기 위한 항상성 비주얼 도메인 어댑터를 제안했습니다.[11]

### 7.2 현재 도전 과제

**의사 레이블 노이즈 완화**: 최신 연구들은 여전히 의사 레이블의 노이즈 문제에 직면하고 있습니다. 불확실성 추정, SAM 유도 신뢰 가능한 의사 레이블, 적응적 가중치 손실 등 다양한 접근이 시도되고 있습니다.[12][13][14]

**모델 붕괴 방지**: 엔트로피 최소화 기반 방법들은 모든 이미지를 단일 클래스로 예측하는 자명한 해로 수렴하는 모델 붕괴 문제를 겪습니다. Ranked entropy, marginal entropy maximization 등의 기법이 제안되었습니다.[8][7]

**클래스 불균형**: 의사 레이블 분포의 불균형 특성이 다중 소스 도메인 적응에서 여전히 문제로 지적되고 있습니다.[15]

### 7.3 미래 연구 방향 및 권고 사항

**일관성 정규화 및 데이터 증강과의 통합**: 논문의 결론에서 언급된 것처럼, 자가 학습 외에 일관성 정규화와 데이터 증강 같은 준지도 학습 기법들을 분포 변화 상황에서 활용하는 연구가 필요합니다.[1]

**대규모 사전 학습 모델과의 결합**: 
- Vision-Language Models (VLMs)에서 CST 원리 적용: CLIP 등의 모델은 사전 학습 데이터의 편향으로 인해 편향된 불확실성 추정을 생성하는데, Adaptive Debiasing Tsallis Entropy (ADTE)가 이를 해결하려는 시도를 보여줍니다.[16]
- Foundation 모델의 도메인 적응: 최신 연구는 foundation 모델에서 source-free 설정으로 확장하고 있습니다.[17]

**점진적 도메인 적응**: Stratified Domain Adaptation (StrDA, 2024)처럼 훈련 데이터를 계층화하여 도메인 간격의 점진적 증가에 적응하는 방법이 주목받고 있습니다.[18][19]

**개인정보 보호 및 지적 재산권 고려**: Source-free 설정이 점점 더 중요해지고 있으며, 소스 데이터 접근 없이 도메인 적응을 수행하는 방법론 개발이 지속될 것입니다.[20][10][9]

**다중 레이블 및 개방 집합 시나리오**: 
- 다중 레이블 TTA에서 Bounded Entropy Minimization (2025)이 제안되었습니다.[21]
- 개방 집합 TTA를 위한 통합 엔트로피 최적화 프레임워크가 개발되었습니다.[22]

**실시간 및 연속 적응**: 실제 환경에서 연속적으로 변화하는 데이터 스트림에 적응하는 온라인 방법론 개발이 중요합니다. Betting martingales와 온라인 학습을 결합한 접근이 시도되고 있습니다.[2]

**이론적 보장 강화**: Expansion 가정을 완화하거나 더 현실적인 가정 하에서 일반화 보장을 제공하는 이론 연구가 필요합니다. 최근 label propagation과 gradual adaptation에 대한 이론적 분석이 진행되고 있습니다.[23][24]

**응용 도메인 확장**: 
- 의료 영상, 원격 감지, 감정 인식, 질의응답 등 실제 응용에서 CST 원리의 효과 검증[25][26][9][17]
- 시계열 데이터와 포인트 클라우드 등 새로운 데이터 모달리티로의 확장[27][28]

### 7.4 연구 시 고려할 실용적 제안

**하이퍼파라미터 자동 조정**: CST의 $$\alpha$$ 자동 선택처럼, 임계값이나 가중치 등의 수동 조정이 필요한 하이퍼파라미터를 자동으로 최적화하는 메커니즘 개발이 중요합니다.

**계산 효율성**: 이중 수준 최적화의 계산 비용을 더욱 줄이기 위한 근사 기법이나 효율적 구현 방법 연구가 필요합니다.

**벤치마크 표준화**: TTA, SFDA, CTTA 등 다양한 설정에서 공정한 비교를 위한 표준화된 벤치마크와 프로토콜 개발이 필요합니다.

**강건성 및 안정성**: 적대적 샘플, 노이즈가 많은 데이터, 분포 외 샘플에 대한 강건성을 높이는 연구가 계속되어야 합니다.[12]

***

Cycle Self-Training은 도메인 적응에서 자가 학습의 근본적인 문제를 해결하는 중요한 이정표입니다. 순환 메커니즘을 통한 의사 레이블 일반화와 Tsallis 엔트로피를 통한 적응적 불확실성 제어라는 핵심 아이디어는 이후 연구에 광범위한 영향을 미치고 있으며, test-time adaptation, source-free adaptation, continual learning 등 다양한 방향으로 확장되고 있습니다. 향후 연구는 대규모 사전 학습 모델과의 통합, 더 강력한 이론적 보장, 그리고 실제 응용에서의 효율성과 강건성 향상에 초점을 맞출 것으로 예상됩니다.[29][30][31][3][16][10][9][7][2][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4cc91091-ab6e-4960-8625-302569e86e35/2103.03571v3.pdf)
[2](https://arxiv.org/html/2408.07511v1)
[3](https://arxiv.org/abs/2308.02746)
[4](https://consensus.app/papers/metatsallisentropy-minimization-a-new-selftraining-lu-huang/5bc0fe5446645530adf74cae92af78b5/)
[5](https://iclr.cc/virtual/2021/spotlight/3479)
[6](https://seohyun00.tistory.com/43)
[7](https://arxiv.org/abs/2505.16441)
[8](https://proceedings.mlr.press/v267/han25e.html)
[9](https://arxiv.org/html/2510.26826v1)
[10](http://arxiv.org/pdf/2212.04227.pdf)
[11](http://arxiv.org/pdf/2306.04344.pdf)
[12](https://arxiv.org/html/2507.00608v1)
[13](https://openaccess.thecvf.com/content/CVPR2023/papers/Litrico_Guiding_Pseudo-Labels_With_Uncertainty_Estimation_for_Source-Free_Unsupervised_Domain_Adaptation_CVPR_2023_paper.pdf)
[14](https://www.sciencedirect.com/science/article/abs/pii/S0925231225014213)
[15](https://www.ijcai.org/proceedings/2024/0516.pdf)
[16](https://openreview.net/pdf/9c3460adbc8d9c6c54eb1cb2369eb85506da029f.pdf)
[17](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00669/121543/Source-Free-Domain-Adaptation-for-Question)
[18](https://arxiv.org/html/2410.09913)
[19](https://arxiv.org/abs/2410.09913)
[20](https://arxiv.org/html/2106.11653v5)
[21](https://openreview.net/forum?id=75PhjtbBdr)
[22](https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_Unified_Entropy_Optimization_for_Open-Set_Test-Time_Adaptation_CVPR_2024_paper.pdf)
[23](https://www.isca-archive.org/interspeech_2025/damianos25_interspeech.pdf)
[24](https://dl.acm.org/doi/10.5555/3524938.3525445)
[25](https://ieeexplore.ieee.org/document/10677431/)
[26](https://www.sciencedirect.com/science/article/pii/S0950705125004150)
[27](https://www.sciencedirect.com/science/article/pii/S0020025524016487)
[28](https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_Self-Supervised_Global-Local_Structure_Modeling_for_Point_Cloud_Domain_Adaptation_With_CVPR_2022_paper.pdf)
[29](https://aclanthology.org/2023.acl-long.92.pdf)
[30](https://arxiv.org/pdf/2104.12928.pdf)
[31](https://arxiv.org/html/2502.06272v1)
[32](https://arxiv.org/pdf/2103.03571.pdf)
[33](https://arxiv.org/pdf/1807.00374.pdf)
[34](http://arxiv.org/pdf/2106.09890.pdf)
[35](https://proceedings.neurips.cc/paper/2021/file/c1fea270c48e8079d8ddf7d06d26ab52-Supplemental.pdf)
[36](https://proceedings.nips.cc/paper/2021/file/c1fea270c48e8079d8ddf7d06d26ab52-Paper.pdf)
[37](https://arxiv.org/pdf/2202.12040.pdf)
[38](https://www.tandfonline.com/doi/full/10.1080/01431161.2025.2450564?ai=179&mi=l49ppp&af=R)
[39](https://arxiv.org/abs/2103.03571)
[40](https://pure.kaist.ac.kr/en/publications/semi-supervised-domain-adaptation-via-selective-pseudo-labeling-a/)
[41](https://onlinelibrary.wiley.com/doi/10.1111/mice.13315?af=R)
[42](https://www.sciencedirect.com/science/article/pii/S266730532400142X)
[43](https://dl.acm.org/doi/10.1145/3593013.3594008)
[44](https://www.semanticscholar.org/paper/Cycle-Self-Training-for-Domain-Adaptation-Liu-Wang/e602ce17a993d33d114381be4dc54e7c19d01bce)
[45](https://arxiv.org/pdf/2410.01709.pdf)
[46](https://arxiv.org/pdf/2303.10856.pdf)
[47](https://arxiv.org/html/2402.06809v2)
[48](https://ieeexplore.ieee.org/document/11177220/)
[49](https://dmqa.korea.ac.kr/uploads/seminar/%5B250207%5DDMQA_Openseminar_Test_Time_Adaptation.pdf)
