# Latent Domain Learning with Dynamic Residual Adapters | Domain adaption, Dynamic Residual Adapters

## 핵심 주장과 주요 기여

이 논문은 **도메인 레이블 없이 다중 도메인에서 학습하는 현실적인 문제**를 다룹니다. 기존의 다중 도메인 학습 방법들이 도메인 주석에 의존하는 반면, 본 연구는 **잠재 도메인 학습(Latent Domain Learning)**이라는 새로운 패러다임을 제시합니다[1][2].

**주요 기여:**
- **Dynamic Residual Adapters (DRA)**: 적응적 게이팅 메커니즘을 통해 잠재 도메인을 자동으로 감지하고 처리
- **Style Exchange Augmentation**: 스타일 전이 기법에서 영감을 얻은 데이터 증강 전략
- **매개변수 효율성**: 기존 방법 대비 현저히 적은 매개변수로 우수한 성능 달성

## 해결하고자 하는 문제

### 문제 정의
표준 딥러닝 모델은 **다중 도메인 데이터에서 큰 도메인에 과적합**되면서 **작은 도메인을 무시하는 경향**을 보입니다. 예를 들어, Visual험에서 단일 ResNet26은 SVHN(40.6% 비중)과 같은 큰 도메인에서는 성능을 유지하지만, Aircraft(5.2% 비중)나 UCF101(1.6% 비중) 같은 작은 도메인에서는 성능이 크게 저하됩니다[1].

### 기존 방법의 한계
- **도메인 주석 의존성**: 기존 다중 도메인 학습은 수동 주석과 데이터셋 큐레이션이 필요
- **확장성 문제**: 도메인별 개별 모델 학습은 매개변수 수가 선형적으로 증가
- **실용성 부족**: 실제 환경에서는 도메인 경계가 명확하지 않음

## 제안하는 방법

### Dynamic Residual Adapters (DRA)

**핵심 수식:**
잠재 도메인을 고려한 ResNet 블록의 변환은 다음과 같이 정의됩니다:

$$
x + f_{\theta_l}(x) + \sum_{k=1}^{K} g_{lk}(x)h_{\alpha_{lk}}(x)
$$

여기서:
- $$f_{\theta_l}(x)$$: 기본 ResNet 컨볼루션
- $$g_{lk}(x)$$: k번째 게이트의 활성화 함수
- $$h_{\alpha_{lk}}(x)$$: k번째 경량 컨볼루션 보정

**게이팅 메커니즘:**

$$
g(x) = \text{Softmax}\{W^T\phi(x) + \varepsilon\}
$$

여기서 $$\phi(x)$$는 채널 차원으로의 평균 풀링, $$\varepsilon \sim \mathcal{N}(0, \Sigma_\varepsilon)$$는 탐색 노이즈입니다[1].

### Style Exchange Augmentation

**수식:**

$$
\eta \left(\frac{\sigma_{a_z}}{\sigma_{a_x}}(a_x - \mu_{a_x}) + \mu_{a_z}\right) + (1-\eta)a_x
$$

이 방법은 서로 다른 샘플 간의 스타일 정보를 교환하여 도메인에 불변한 표현 학습을 촉진합니다[1].

## 모델 구조

DRA는 **기존 ResNet 아키텍처에 매끄럽게 통합**될 수 있는 구조로 설계되었습니다:

1. **고정된 백본**: ImageNet에서 사전 훈련된 ResNet 매개변수는 고정
2. **동적 게이트**: 각 층에서 K개의 전문가 중 선택하는 소프트 어텐션 메커니즘
3. **경량 어댑터**: 1×1 컨볼루션을 통한 매개변수 효율적 보정

## 성능 향상

### 정량적 결과

**Visual Decathlon 실험:**
| 모델 | 매개변수 | Aircraft | CIFAR-100 | DTD | UCF101 | 평균 |
|------|----------|----------|-----------|-----|--------|------|
| 9×ResNet26 | ~55.8M | 39.48 | 77.96 | 38.19 | 73.00 | 87.01 |
| ResNet26 | 6.2M | 31.35 | 70.71 | 33.67 | 58.25 | 84.73 |
| DRA | ~1.4M | 38.28 | 78.16 | 40.64 | 63.88 | 86.46 |

**PACS 데이터셋 결과:**
| 모델 | 매개변수 | Art Painting | Cartoon | Photo | Sketch | 평균 |
|------|----------|--------------|---------|-------|--------|------|
| 4×ResNet26 | 24.8M | 88.77 | 95.97 | 95.95 | 95.83 | 94.44 |
| DRA K=2 | 1.4M | 92.15 | 96.62 | 97.09 | 95.34 | 95.28 |

### 주요 성과
- **매개변수 효율성**: 기존 방법 대비 **90% 적은 매개변수**로 더 나은 성능
- **작은 도메인 보호**: 큰 도메인에 치우치지 않고 모든 도메인에서 균형잡힌 성능
- **일반화 능력**: 도메인 레이블이 있는 지도 학습 방법보다 우수한 성능

## 모델의 일반화 성능 향상

### 동적 적응 메커니즘
DRA의 핵심은 **입력별 동적 매개변수 공유**입니다. 게이트 활성화 경로 분석 결과, 시각적으로 유사한 도메인(예: art painting과 photo)은 유사한 활성화 패턴을 보이며, 이는 **의미 있는 도메인 클러스터링**이 자동으로 이루어짐을 시사합니다[1].

### 크로스 도메인 지식 공유
전통적인 도메인별 모델과 달리, DRA는 **도메인 간 매개변수 공유를 통한 지식 전이**를 가능하게 합니다. 이는 특히 작은 도메인에서 큰 성능 향상을 가져왔습니다.

### 강건성
- **잘못된 도메인 주석에 대한 내성**: PACS 데이터셋에서 인간이 잘못 라벨링한 샘플도 올바르게 분류
- **노이즈가 있는 게이팅**: 게이트 예측이 부정확해도 여러 전문가의 가중 조합으로 안정적 성능 유지

## 한계

### 방법론적 한계
- **하이퍼파라미터 민감도**: K(전문가 수)와 η(증강 강도) 선택이 성능에 중요한 영향
- **컴퓨팅 오버헤드**: 다중 전문가 평가로 인한 추가 연산 비용
- **해석 가능성**: 게이트 활성화 패턴의 의미론적 해석에 한계

### 실험적 한계
- **제한된 도메인 범위**: 주로 이미지 분류 태스크에 국한된 검증
- **스케일 문제**: 매우 많은 수의 잠재 도메인에 대한 확장성 미검증

## 미래 연구에 미치는 영향

### 긍정적 영향
1. **실용적 패러다임 전환**: 도메인 주석 없는 학습의 중요성 부각
2. **매개변수 효율적 적응**: 제한된 자원 환경에서의 모델 적응 연구 촉진
3. **동적 아키텍처**: 입력에 따른 동적 네트워크 구조 변경 연구 확산

### 후속 연구 방향
- **자연어 처리 적용**: 텍스트 도메인에서의 잠재 도메인 학습[3]
- **의료 영상 분야**: 병원별, 장비별 도메인 차이 해결[3]
- **연속 학습**: 시간에 따른 도메인 변화 적응[4]

## 앞으로 연구 시 고려할 점

### 기술적 고려사항
1. **확장성**: 수백 개 이상의 잠재 도메인 처리 방법 연구 필요
2. **효율성**: 실시간 응용을 위한 계산 복잡도 최적화
3. **안정성**: 도메인 분포 변화에 대한 강건성 확보

### 응용 고려사항
1. **도메인 발견**: 자동 도메인 수 결정 및 경계 탐지 알고리즘 개발
2. **멀티모달 확장**: 이미지-텍스트, 오디오-비주얼 등 다중 모달리티 처리
3. **페어니스**: 알고리즘 공정성 관점에서의 소수 도메인 보호 메커니즘 강화

### 평가 방법론
1. **새로운 메트릭**: 잠재 도메인 학습에 특화된 평가 지표 개발[5]
2. **벤치마크**: 다양한 도메인 불균형 시나리오를 포함한 표준 데이터셋 구축
3. **해석 가능성**: 모델 결정 과정의 투명성 확보 방안

이 연구는 **실제 환경에서의 모델 적응**이라는 중요한 문제를 제기하며, 도메인 적응 연구 분야에 새로운 방향을 제시했습니다. 특히 매개변수 효율성과 일반화 성능의 균형을 이루는 혁신적 접근법으로, 향후 실용적 AI 시스템 개발에 중요한 기여를 할 것으로 예상됩니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f7f2b87a-728d-4281-8613-a4e60beec6d0/2006.00996v1.pdf
[2] https://www.semanticscholar.org/paper/9c0f5f07997b439d78956b105a792a75a0285a1c
[3] https://arxiv.org/abs/2401.03002
[4] https://openaccess.thecvf.com/content/WACV2025/papers/Cheng_DAM_Dynamic_Adapter_Merging_for_Continual_Video_QA_Learning_WACV_2025_paper.pdf
[5] https://openreview.net/forum?id=fszrlQ2DuP
[6] https://linkinghub.elsevier.com/retrieve/pii/S0020025524001361
[7] https://ieeexplore.ieee.org/document/10318150/
[8] https://ieeexplore.ieee.org/document/9578733/
[9] https://ieeexplore.ieee.org/document/9831691/
[10] https://ieeexplore.ieee.org/document/10831990/
[11] https://ieeexplore.ieee.org/document/9905733/
[12] https://pubmed.ncbi.nlm.nih.gov/31251207/
[13] https://www.isca-archive.org/interspeech_2023/mehrish23_interspeech.pdf
[14] https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Model_Adaptation_Unsupervised_Domain_Adaptation_Without_Source_Data_CVPR_2020_paper.pdf
[15] https://pubmed.ncbi.nlm.nih.gov/31398109/
[16] https://arxiv.org/abs/2006.00996
[17] https://arxiv.org/pdf/2006.00996.pdf
[18] https://arxiv.org/pdf/1901.05335.pdf
[19] https://groups.inf.ed.ac.uk/vico/assets/pdf/Deecke22.pdf
[20] https://www.mdpi.com/2078-2489/16/6/483
[21] https://arxiv.org/abs/1901.05335
[22] https://openreview.net/forum?id=McYsRk9-rso
[23] https://arxiv.org/pdf/2407.06204.pdf
[24] https://paperswithcode.com/task/unsupervised-domain-adaptation
[25] https://arxiv.org/html/2207.07624v2
[26] https://www.semanticscholar.org/paper/45b932394eb565c18c2d8043721e79b478ae38f1
[27] https://aclanthology.org/2021.emnlp-demo.27
[28] https://arxiv.org/abs/2308.14596
[29] http://arxiv.org/pdf/1907.00953.pdf
[30] https://arxiv.org/pdf/2208.03345.pdf
[31] https://arxiv.org/pdf/2210.03728.pdf
[32] https://arxiv.org/pdf/2212.04065.pdf
[33] http://arxiv.org/pdf/2009.07044.pdf
[34] https://dl.acm.org/doi/pdf/10.1145/3597503.3639106
[35] http://arxiv.org/pdf/2502.12128.pdf
[36] https://arxiv.org/pdf/2311.06816.pdf
[37] https://github.com/xialeiliu/Awesome-Incremental-Learning
[38] https://www.themoonlight.io/ko/review/model-adaptation-unsupervised-domain-adaptation-without-source-data
[39] https://openaccess.thecvf.com/content_cvpr_2018/papers/Mancini_Boosting_Domain_Adaptation_CVPR_2018_paper.pdf
[40] https://icml.cc/virtual/2025/poster/43454
