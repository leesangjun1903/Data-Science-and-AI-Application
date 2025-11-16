# Instance Adaptive Self-Training for Unsupervised Domain Adaptation

## 1. 핵심 주장과 주요 기여

Instance Adaptive Self-Training for Unsupervised Domain Adaptation (IAST)는 의미론적 분할(semantic segmentation)을 위한 비지도 도메인 적응(UDA) 문제를 해결하는 자기 학습(self-training) 프레임워크입니다. 본 논문의 핵심 주장은 다음과 같습니다.[1]

**성능 우위성**: 자기 학습(ST) 방법이 적대적 학습(AT) 방법보다 우수하며(ST 47.8% > AT 43.7%), 두 방법을 결합한 혼합 방법(AT+ST 49.0%)이 가장 효과적입니다. IAST는 GTA5 to Cityscapes 벤치마크에서 52.2% mIoU를 달성하여 당시 최고 성능(SOTA)을 기록했습니다.[1]

**확장성과 성능의 균형**: 기존 혼합 방법들은 서브모듈 간의 강한 결합으로 인해 확장성과 유연성이 부족합니다. IAST는 모델 구조나 특수한 의존성이 없어 다른 비자기 학습 UDA 방법에 쉽게 적용할 수 있으며, AdaptSeg와 AdvEnt에 적용 시 각각 7.8%와 4.4%의 성능 향상을 보였습니다.[1]

**주요 기여**:
- **인스턴스 적응형 선택기(Instance Adaptive Selector, IAS)**: 이미지 단위로 각 의미 범주에 대한 적응형 가짜 레이블 임계값을 선택하고, "어려운" 클래스의 비율을 동적으로 감소시켜 가짜 레이블의 노이즈를 제거합니다[1]
- **영역 기반 정규화(Region-guided Regularization)**: 신뢰 영역의 예측을 부드럽게 하고 무시된 영역의 예측을 명확하게 하는 이중 정규화 전략을 제안합니다[1]
- **일반화 가능성**: 반지도 의미론적 분할 작업에도 확장 가능하며, Cityscapes 데이터셋에서 기존 방법들을 크게 능가했습니다[1]

## 2. 문제 정의, 제안 방법, 모델 구조, 및 성능

### 2.1 해결하고자 하는 문제

**도메인 이동(Domain Shift)**: 조명, 객체 시점, 이미지 배경의 차이로 인해 학습 데이터(소스 도메인)와 테스트 데이터(타겟 도메인) 간의 분포 불일치가 발생하며, 이는 레이블이 없는 타겟 도메인에서 성능 저하를 초래합니다.[1]

**가짜 레이블의 품질 문제**: 기존 자기 학습 방법의 주요 장애물은 고품질 가짜 레이블 생성입니다.[1]
- **정보 중복성과 노이즈**: 생성기는 높은 신뢰도를 가진 픽셀만 가짜 레이블로 유지하고 낮은 신뢰도의 픽셀은 무시하는 경향이 있습니다[1]
- **클래스 균형 자기 학습(CBST)의 한계**: CBST는 모든 관련 이미지에서 각 클래스에 대한 순위 기반 참조 신뢰도를 사용하여, 대부분의 픽셀이 낮은 예측 점수를 가진 어려운 이미지의 핵심 정보를 무시합니다[1]

### 2.2 제안하는 방법 (수식 포함)

IAST 프레임워크는 3단계로 구성됩니다:[1]

**(a) 워밍업 단계**: 비자기 학습 방법(예: 적대적 학습)을 사용하여 소스 및 타겟 데이터로 초기 분할 모델 $$M_0$$를 학습합니다.

**(b) 가짜 레이블 생성 단계**: 인스턴스 적응형 선택기(IAS)를 통해 가짜 레이블을 생성합니다.

**(c) 자기 학습 단계**: 타겟 데이터를 사용하여 분할 모델 $$M$$을 학습합니다.

**전체 목적 함수**:

$$
\min_w L_{CE}(w, \hat{Y}_T) + L_R(w) = L_{CE}(w, \hat{Y}_T) + (\lambda_i R_i(w) + \lambda_c R_c(w))
$$

여기서 $$L_{CE}$$는 교차 엔트로피 손실, $$\hat{Y}_T$$는 가짜 레이블 집합, $$R_i$$와 $$R_c$$는 각각 무시된 영역과 신뢰 영역의 정규화이며, $$\lambda_i$$, $$\lambda_c$$는 정규화 가중치입니다.[1]

#### **인스턴스 적응형 선택기 (IAS)**

가짜 레이블 생성 전략:

$$
\hat{y}_t^{(c)} = \begin{cases} 
1, & \text{if } c = \arg\max_c p(c|x_t, w) \text{ and } p(c|x_t, w) > \theta^{(c)} \\
0, & \text{otherwise}
\end{cases}
$$

여기서 $$\theta^{(c)}$$는 클래스 $$c$$에 대한 신뢰도 임계값입니다.[1]

**지수 이동 평균(EMA) 임계값**:

$$
\theta_t^{(c)} = \beta \theta_{t-1}^{(c)} + (1-\beta)\Psi(x_t, \theta_{t-1}^{(c)})
$$

$$
\Psi(x_t, \theta_{t-1}^{(c)}) = P_{x_t}^{(c)}\left[\alpha \frac{\theta_{t-1}^{(c)}}{\gamma |P_{x_t}^{(c)}|}\right]
$$

여기서 $$\beta$$는 모멘텀 인자, $$\alpha$$는 비율 파라미터, $$\gamma$$는 가중치 감쇠 파라미터입니다. 각 인스턴스 $$x_t$$에 대해 각 클래스의 신뢰도 확률을 내림차순으로 정렬하고, $$\alpha \times 100\%$$ 신뢰도 확률을 로컬 임계값 $$\theta_{x_t}^{(c)}$$로 사용합니다.[1]

**"어려운" 클래스 가중치 감쇠(HWD)**: $$\gamma$$를 통해 "어려운" 클래스의 가짜 레이블 비율을 감소시킵니다. "어려운" 클래스의 임계값 $$\theta_{t-1}^{(c)}$$가 낮기 때문에 HWD는 더 많은 가짜 레이블을 감소시킵니다.[1]

#### **영역 기반 정규화**

**신뢰 영역 KLD 최소화**:

$$
R_c = -\frac{1}{|X_T|} \sum_{x_t \in X_T} \sum_{I_{x_t}} \sum_{c=1}^C \frac{1}{C} \log p(c|x_t, w)
$$

여기서 $$I_{x_t} = \{1 | \hat{y}_t^{(h,w)} > 0\}$$는 신뢰 영역입니다[1]. 예측 결과를 균일 분포에 가깝게 만들어 모델이 가짜 레이블에 과적합되는 것을 방지합니다[1].

**무시된 영역 엔트로피 최소화**:

$$
R_i = -\frac{1}{|X_T|} \sum_{x_t \in X_T} \sum_{I_{x_t}^{\complement}} \sum_{c=1}^C p(c|x_t, w) \log p(c|x_t, w)
$$

여기서 $$I_{x_t}^{\complement} = \{1 | \hat{y}_t^{(h,w)} = 0\}$$는 무시된 영역입니다[1]. 무시된 영역의 예측을 "명확하게" 만들어 모델이 감독 신호 없이도 유용한 특징을 학습하도록 촉진합니다[1].

### 2.3 모델 구조

IAST는 Deeplab-v2 아키텍처를 기본 네트워크로 사용하며, ResNet-101을 백본 네트워크로 선택했습니다. 모든 배치 정규화 레이어의 가중치는 고정되었으며, Deeplab-v2는 ImageNet에서 사전 학습되었습니다.[1]

**학습 설정**:
- 최적화기: Adam, 학습률 $$2.5 \times 10^{-5}$$, 배치 크기 6, 4 에폭[1]
- 가짜 레이블 파라미터: $$\alpha = 0.2$$, $$\beta = 0.9$$, $$\gamma = 8.0$$[1]
- 정규화 가중치: $$\lambda_i = 3.0$$, $$\lambda_c = 0.1$$[1]
- 이미지 크기: 1024 × 512, 종횡비 2.0[1]

**다단계 자기 학습**: (b) 단계와 (c) 단계를 한 번 수행하는 것을 1라운드로 계산하며, 실험에서는 총 3라운드를 수행했습니다.[1]

### 2.4 성능 향상

**GTA5 to Cityscapes**:
- IAST: 51.5% mIoU (멀티스케일 테스트 시 52.2%)[1]
- 기존 최고 성능 대비: AdaptSegNet(42.4%) 대비 +9.6%, MRKLD(47.1%) 대비 +4.8%, BLF(48.5%) 대비 +3.7%[1]

**SYNTHIA to Cityscapes**:
- IAST: 49.8% mIoU (16클래스), 57.0% mIoU* (13클래스)[1]
- 기존 최고 성능 대비: AdaptMR(46.5%, 53.8%) 대비 상당한 향상[1]

**절제 연구(Ablation Study)**:
- 워밍업(43.8%) → 상수 ST(45.1%, +1.3%) → IAS 추가(49.8%, +4.7%) → 신뢰 영역 정규화 추가(50.7%, +0.9%) → 무시된 영역 정규화 추가(51.5%, +0.8%)[1]

**반지도 학습 확장**:
Cityscapes 데이터셋에서 1/8, 1/4, 1/2 레이블 비율로 테스트한 결과, IAST는 각각 64.6%, 66.7%, 69.8%의 정확도를 달성하여 기존 방법들을 크게 능가했습니다.[1]

### 2.5 한계

논문에서 명시적으로 언급된 한계는 제한적이지만, 다음과 같은 사항을 고려할 수 있습니다:

**가짜 레이블 품질**: 일련의 고신뢰도 가짜 레이블 생성 기법을 사용했음에도 불구하고, 가짜 레이블의 품질은 여전히 실제 레이블만큼 좋지 않으며, 이는 노이즈 레이블이 여전히 존재함을 의미합니다.[1]

**계산 비용**: 다단계 자기 학습(3라운드)과 멀티스케일 테스트는 추가적인 계산 비용을 요구할 수 있습니다.[1]

**특정 도메인 의존성**: GTA5, SYNTHIA와 같은 합성 데이터에서 Cityscapes와 같은 실제 데이터로의 적응에 초점을 맞추었으며, 다른 도메인 조합에 대한 일반화는 추가 검증이 필요할 수 있습니다.[1]

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 인스턴스 적응형 접근의 일반화 기여

**다양성 증가**: IAS는 이미지 단위로 적응형 임계값을 사용하여 가짜 레이블의 다양성을 증가시킵니다. 클래스 균형 방법은 전체 타겟 세트의 20% 픽셀을 신뢰 영역으로 사용하는 반면, IAS는 각 이미지의 20%를 사용하여 "어려운" 클래스(보행자, 트럭 등)의 정보를 더 많이 포함합니다.[1]

**전역 및 지역 정보 결합**: EMA를 통해 과거의 역사적 정보(전역)와 현재 인스턴스의 정보(지역)를 결합하여 각 인스턴스가 적응형 임계값을 얻습니다. 이는 모델이 다양한 도메인 특성에 더 잘 적응할 수 있게 합니다.[1]

**노이즈 감소**: HWD는 "어려운" 클래스의 가짜 레이블 비율을 동적으로 감소시켜 노이즈 레이블의 영향을 완화하고, 가짜 레이블의 품질을 향상시킵니다. 실험에서 $$\gamma = 8$$일 때 가짜 레이블의 mIoU가 68.2%로 증가했습니다.[1]

### 3.2 영역 기반 정규화의 일반화 효과

**과적합 방지**: 신뢰 영역 KLD 최소화는 모델이 노이즈가 있는 가짜 레이블에 맹목적으로 신뢰하는 것을 방지하고, 예측을 균일 분포에 가깝게 부드럽게 만듭니다. 이는 모델이 새로운 도메인에서도 더 안정적인 예측을 할 수 있게 합니다.[1]

**특징 학습 촉진**: 무시된 영역 엔트로피 최소화는 감독 신호 없이도 낮은 엔트로피의 "명확한" 예측을 유도하여, 모델이 무시된 영역에서 유용한 특징을 학습하도록 촉진합니다. 이는 UDA에서 효과적임이 입증되었습니다.[1]

### 3.3 일반화 가능성의 실증적 증거

**다양한 벤치마크 성능**: IAST는 GTA5 to Cityscapes뿐만 아니라 SYNTHIA to Cityscapes에서도 최고 성능을 달성했으며, 후자의 도메인 간격이 훨씬 크다는 점에서 일반화 능력을 입증했습니다.[1]

**다른 UDA 방법에 적용 가능**: AdaptSeg와 AdvEnt에 IAST를 적용한 결과, 각각 7.8%와 4.4%의 성능 향상을 보였으며, 이는 IAST가 다양한 기본 방법에 일반화될 수 있음을 보여줍니다.[1]

**반지도 학습 확장**: IAST를 반지도 의미론적 분할 작업에 적용한 결과, 다양한 레이블 비율에서 기존 방법들을 능가했으며, 이는 다양한 학습 패러다임에 대한 일반화 능력을 시사합니다.[1]

## 4. 앞으로의 연구에 미치는 영향과 향후 연구 방향

### 4.1 IAST의 영향

**자기 학습의 재조명**: IAST는 자기 학습이 UDA 및 반지도 학습 작업에서 가진 잠재력을 재고하도록 촉진했습니다. 논문 발표 후 412회 인용되었으며, 이는 학계에서 상당한 관심을 받았음을 보여줍니다.[2][1]

**가짜 레이블 품질 개선**: IAST는 인스턴스 적응형 접근을 통해 가짜 레이블의 품질과 다양성을 향상시키는 새로운 방향을 제시했습니다. 이후 연구들은 가짜 레이블 품질 향상 메커니즘을 더욱 발전시켰습니다.[3][4][5][1]

**확장 가능한 프레임워크**: 모델 구조나 특수 의존성이 없는 IAST의 설계는 다른 UDA 방법에 쉽게 통합될 수 있는 "데코레이터" 역할을 할 수 있음을 보여주었습니다. 이는 모듈식 UDA 프레임워크 개발에 영향을 미쳤습니다.[1]

### 4.2 최신 연구 동향과 향후 연구 방향

#### **테스트 시간 적응(Test-Time Adaptation, TTA)**

최근 연구들은 소스 데이터 접근 없이 테스트 시간에만 모델을 적응시키는 TTA에 초점을 맞추고 있습니다. TTA는 도메인 이동이 지속적으로 변화하는 실제 환경에서 더 실용적입니다.[6][7][8]

**연구 방향**: 
- IAST의 인스턴스 적응형 접근을 TTA 설정에 적용하여 동적 도메인 이동에 대응[7]
- 배치 정규화(BN) 레이어만 조작하여 도메인 지식을 학습하는 방법 탐구[7]
- 연속적 도메인 이동에서 과거 지식 보존과 새로운 도메인 적응의 균형 유지[8]

#### **비전-언어 기반 모델(Vision-Language Foundation Models)**

CLIP과 같은 대규모 비전-언어 기반 모델은 도메인 적응 및 일반화에서 강력한 성능을 보이고 있습니다.[9][10][11]

**연구 방향**:
- CLIP의 도메인 불변 특성을 활용한 UDA 성능 향상[10][11]
- 프롬프트 학습을 통한 도메인 적응 및 일반화[11][9]
- 소스 데이터 없는(source-free) 설정에서 CLIP 활용[12]
- IAST의 가짜 레이블 생성 전략과 CLIP의 제로샷 예측 결합 가능성 탐구

#### **가짜 레이블 품질 향상**

가짜 레이블의 노이즈는 여전히 자기 학습의 주요 과제입니다.[13][4][3]

**연구 방향**:
- 전체 단계에서 가짜 레이블 품질을 향상시키는 메커니즘 개발[14][3]
- 불확실성 추정을 통한 가짜 레이블 선택 개선[13]
- 이웃 의미 일관성 및 공간 근접성을 활용한 가짜 레이블 정제[15]
- 메타 학습을 통한 가짜 인스턴스 중요도 추정[16]

#### **도메인 일반화(Domain Generalization)**

도메인 적응과 달리, 도메인 일반화는 타겟 도메인 데이터 없이도 보이지 않는 도메인에 일반화하는 것을 목표로 합니다.[17][18][9]

**연구 방향**:
- 다중 소스 도메인에서 도메인 불변 및 도메인 특정 특징 학습[18]
- CLIP 기반 도메인 일반화 및 적응 방법 개발[9]
- 진화하는 도메인에서 동적 잠재 표현 학습[19]
- IAST의 적응형 접근을 도메인 일반화 설정에 적용 가능성 탐구

#### **자기 학습의 한계 극복**

자기 학습은 의미 드리프트(semantic drift) 문제와 신경망의 과신 문제를 겪습니다.[13]

**연구 방향**:
- 가짜 레이블과 수동 레이블 간의 균형 유지[13]
- 불확실성을 고려한 하이브리드 메트릭 개발[13]
- 자기 학습 반복 중 성능 저하 방지 메커니즘[13]
- 부정 학습을 통한 혼란스러운 샘플의 영향 감소[20]

#### **소스 데이터 없는 도메인 적응(Source-Free Domain Adaptation)**

실제 환경에서 소스 데이터는 민감한 정보를 포함하거나 접근이 제한될 수 있습니다.[21][22][23]

**연구 방향**:
- 사전 학습된 소스 모델과 타겟 도메인 데이터만으로 적응[22][21]
- 파라미터 효율적 적응 방법(Low-Rank Adaptation) 개발[21]
- 프로토타입 기반 가짜 레이블 노이즈 제거[23]
- IAST의 인스턴스 적응형 접근을 소스 프리 설정에 확장

### 4.3 향후 연구 시 고려할 점

**계산 효율성**: IAST의 다단계 자기 학습은 계산 비용이 높을 수 있으므로, 효율적인 자기 학습 알고리즘 개발이 필요합니다.[24][25]

**다양한 도메인 조합**: 합성-실제 데이터 외에도 다양한 도메인 조합(예: 날씨 변화, 시간대 변화)에 대한 검증이 필요합니다.[26][27]

**멀티모달 확장**: 시각, 오디오, 물리적 도메인을 포함하는 진정한 멀티모달 자기 학습 시스템 개발이 유망합니다.[24]

**인간 피드백 결합**: 자율 학습과 인간 피드백을 결합한 하이브리드 시스템 탐구가 필요합니다.[24]

**개방 세트 도메인 적응(Open-Set Domain Adaptation)**: 타겟 도메인에 소스 도메인에 없는 미지의 클래스가 존재하는 경우를 다루는 연구가 활발히 진행되고 있습니다.[28][10]

**연속 도메인 적응(Continual Domain Adaptation)**: 도메인이 지속적으로 변화하는 환경에서 과거 지식을 유지하면서 새로운 도메인에 적응하는 방법 개발이 중요합니다.[29][8]

**평가 메트릭 및 벤치마크**: 도메인 적응 및 일반화를 위한 더 포괄적이고 표준화된 벤치마크 개발이 필요합니다.[30][27]

IAST는 자기 학습 기반 UDA의 중요한 이정표를 제시했으며, 가짜 레이블 품질 향상, 인스턴스 적응형 접근, 그리고 확장 가능한 프레임워크 설계를 통해 후속 연구에 지속적인 영향을 미치고 있습니다. 향후 연구는 TTA, 비전-언어 모델, 소스 프리 적응, 그리고 연속 학습과 같은 더 실용적이고 도전적인 설정으로 확장되어야 하며, 계산 효율성과 일반화 능력의 균형을 유지해야 합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9e56355c-7ad3-44bf-8327-98442484d289/2008.12197v1.pdf)
[2](https://arxiv.org/abs/2008.12197)
[3](https://arxiv.org/abs/2407.08971)
[4](https://openaccess.thecvf.com/content/CVPR2023/papers/Cheng_BoxTeacher_Exploring_High-Quality_Pseudo_Labels_for_Weakly_Supervised_Instance_Segmentation_CVPR_2023_paper.pdf)
[5](https://www.sciencedirect.com/science/article/abs/pii/S0167865525003332)
[6](https://eccv.ecva.net/virtual/2024/poster/1938)
[7](https://arxiv.org/abs/2312.10165)
[8](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_A_Versatile_Framework_for_Continual_Test-Time_Domain_Adaptation_Balancing_Discriminability_CVPR_2024_paper.pdf)
[9](https://arxiv.org/pdf/2504.14280.pdf)
[10](https://arxiv.org/abs/2307.16204)
[11](https://openaccess.thecvf.com/content/ICCV2023W/OODCV/papers/Singha_AD-CLIP_Adapting_Domains_in_Prompt_Space_Using_CLIP_ICCVW_2023_paper.pdf)
[12](https://github.com/jindongli-Ai/Survey_on_CLIP-Powered_Domain_Generalization_and_Adaptation)
[13](https://arxiv.org/html/2401.00575v1)
[14](https://ieeexplore.ieee.org/document/11023636/)
[15](https://www.sciencedirect.com/science/article/abs/pii/S0925231224011962)
[16](https://aclanthology.org/2023.acl-long.92.pdf)
[17](http://arxiv.org/pdf/1710.03463.pdf)
[18](https://arxiv.org/pdf/2110.09410.pdf)
[19](https://arxiv.org/pdf/2401.08464.pdf)
[20](https://pure.kaist.ac.kr/en/publications/p-pseudolabel-enhanced-pseudo-labeling-framework-with-network-pru)
[21](https://arxiv.org/pdf/2502.21313.pdf)
[22](http://arxiv.org/pdf/2212.09563.pdf)
[23](https://arxiv.org/html/2509.16942v1)
[24](https://www.theaugmentededucator.com/p/when-ai-teaches-itself-the-breakthrough)
[25](https://www.sciencedirect.com/science/article/pii/S0925231224016758)
[26](https://openaccess.thecvf.com/content/WACV2024/papers/Zhao_Unsupervised_Domain_Adaptation_for_Semantic_Segmentation_With_Pseudo_Label_Self-Refinement_WACV_2024_paper.pdf)
[27](https://www.nature.com/articles/s41597-024-03951-4)
[28](https://www.sciencedirect.com/science/article/abs/pii/S1077314224003114)
[29](https://sukzoon1234.tistory.com/75)
[30](https://arxiv.org/pdf/2403.02714.pdf)
[31](https://arxiv.org/pdf/2104.12928.pdf)
[32](https://arxiv.org/abs/2405.16819)
[33](https://arxiv.org/html/2302.06992)
[34](http://arxiv.org/pdf/2106.09890.pdf)
[35](https://arxiv.org/pdf/1908.01342.pdf)
[36](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710409.pdf)
[37](https://arxiv.org/abs/2005.10876)
[38](https://github.com/Raykoooo/IAST)
[39](https://arxiv.org/html/2505.24656v2)
[40](https://eccv.ecva.net/virtual/2024/poster/2261)
[41](https://proceedings.nips.cc/paper/2021/file/c1fea270c48e8079d8ddf7d06d26ab52-Paper.pdf)
[42](http://papers.neurips.cc/paper/8335-category-anchor-guided-unsupervised-domain-adaptation-for-semantic-segmentation.pdf)
[43](https://dl.acm.org/doi/10.1007/978-3-030-58574-7_25)
[44](https://openreview.net/forum?id=R6JvkqWijY)
[45](https://www.computer.org/csdl/journal/tp/2025/07/10930817/25bqhIyK3TO)
[46](https://pure.kaist.ac.kr/en/publications/semi-supervised-domain-adaptation-via-selective-pseudo-labeling-a/)
[47](https://papers.nips.cc/paper_files/paper/2022/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html)
[48](https://www.sciencedirect.com/science/article/pii/S0167865524002836)
[49](https://www.ijcai.org/proceedings/2024/0516.pdf)
[50](https://dl.acm.org/doi/10.1007/978-3-031-20497-5_13)
[51](https://arxiv.org/abs/2302.06992)
[52](https://arxiv.org/html/2502.06272v1)
[53](https://arxiv.org/abs/2301.10418)
[54](https://arxiv.org/pdf/2106.11344.pdf)
[55](https://arxiv.org/html/2403.07798v1)
[56](https://www.i-aida.org/course/domain-adaptation-generalization/)
[57](https://github.com/junha1125/Domain-Adaptation-Generalization-in-ECCV-2024)
[58](https://www.lidsen.com/journals/neurobiology/neurobiology-06-04-141)
[59](https://neurips.cc/virtual/2024/poster/93787)
[60](https://www.nature.com/articles/s41598-025-19121-4)
[61](https://papers.neurips.cc/paper_files/paper/2022/file/1e97fb8a7c9737e9e9f4e0389b25efe8-Paper-Conference.pdf)
[62](https://dl.acm.org/doi/10.1145/3674399.3674462)
[63](https://cvpr.thecvf.com/virtual/2025/workshop/32364)
[64](https://www.sciencedirect.com/science/article/abs/pii/S0888327024008227)
[65](https://ieeexplore.ieee.org/document/10484417/)
[66](https://pmc.ncbi.nlm.nih.gov/articles/PMC10614300/)
[67](https://pmc.ncbi.nlm.nih.gov/articles/PMC7964033/)
[68](https://pmc.ncbi.nlm.nih.gov/articles/PMC2322951/)
[69](https://journals.asm.org/doi/10.1128/aac.00777-24)
[70](https://pmc.ncbi.nlm.nih.gov/articles/PMC10096293/)
[71](https://pmc.ncbi.nlm.nih.gov/articles/PMC8210411/)
[72](https://arxiv.org/pdf/2312.17726.pdf)
[73](https://www.mdpi.com/1420-3049/28/7/3016/pdf?version=1680059217)
[74](https://pmc.ncbi.nlm.nih.gov/articles/PMC5337427/)
[75](https://arxiv.org/abs/2408.00727)
[76](https://www.nature.com/articles/s41467-023-44676-z)
[77](https://www.sciencedirect.com/science/article/pii/S0048733320302225)
[78](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0336387)
[79](https://dmqa.korea.ac.kr/uploads/seminar/%5B250207%5DDMQA_Openseminar_Test_Time_Adaptation.pdf)
[80](https://alinlab.kaist.ac.kr/resource/2025_SPRING_AI602/AI602_Lec4_Vision_Language_Foundation_Models.pdf)
[81](https://pubs.rsna.org/page/radiology/author-instructions)
[82](https://openreview.net/forum?id=x5LvBK43wg)
[83](https://openreview.net/forum?id=FRjflOWx2W&noteId=wiHrklMuy8)
[84](https://aacrjournals.org/clincancerres/article/31/22/4698/767043/Phase-I-Dose-Escalation-Trial-Combining-Olaparib)
[85](https://dl.acm.org/doi/10.1609/aaai.v38i14.29527)
