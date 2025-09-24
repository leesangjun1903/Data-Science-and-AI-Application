# Incremental Multi-domain Learning with Network Latent Tensor Factorization | Domain adaption

## 핵심 주장과 주요 기여

본 논문은 **네트워크 잠재 텐서 인수분해(Network Latent Tensor Factorization)**를 활용한 점진적 다중 도메인 학습 방법을 제안한다[1]. 핵심 주장은 기존의 layer-wise adaptation 방법들과는 달리, CNN의 동일한 구조를 가진 블록들을 하나의 고차원 텐서로 그룹화하여 **훨씬 더 압축적인 표현**을 달성할 수 있다는 것이다[1].

**주요 기여:**
- **7.5배 매개변수 감소**: 새로운 작으로 7.5배 적은 매개변수로 경쟁력 있는 성능 달성[1]
- **텐서 구조 활용**: 단순한 행렬 연산보다 텐서 구조를 활용하는 것이 더 나은 성능을 보여줌[1]
- **레이어 간 상관관계 모델링**: 공동 텐서 모델링을 통해 서로 다른 레이어 간의 상관관계를 자연스럽게 활용[1]

## 해결하고자 하는 문제

### 문제 정의
**점진적 다중 도메인 학습(Incremental Multi-domain Learning)**에서 마주하는 세 가지 주요 문제[1]:

1. **도메인 차이**: 새로운 도메인과 작업이 기존 것과 매우 다를 수 있음
2. **데이터 부족**: 새로운 도메인에서 제한된 양의 주석된 데이터만 사용 가능
3. **계산 비용**: 각 새로운 작업에 대해 전체 모델을 다시 훈련하는 것은 메모리와 계산 측면에서 비현실적

### 기존 방법의 한계
기존의 adapter 기반 방법들은 **layer-specific adaptation modules**를 사용하여 각 CNN 레이어에 개별적으로 적용되지만, 레이어 수에 비례하여 매개변수가 증가하는 문제가 있다[1][2][3]. 실제로 기존 adaptation 네트워크는 약 10%의 추가 매개변수가 필요하다[1].

## 제안하는 방법 (수식 포함)

### 1. 잠재 네트워크 매개변수화 (Latent Network Parametrization)

**6차 텐서 구성**: 각 매크로 모듈에 대해 다음과 같은 6차 텐서를 구성한다[1]:

$$ θ^{(b)} ∈ R^{D_0×D_1×...×D_5} $$

여기서:
- $$D_0×D_1×D_2×D_3$$: 특정 convolution layer의 가중치 형태 (출력 채널, 입력 채널, 커널 폭, 커널 높이)
- $$D_4$$: residual module당 basic block 수 (2개)
- $$D_5$$: 각 매크로 모듈의 residual block 수 (4개)

### 2. Tucker 분해를 이용한 다중 도메인 학습

**소스 도메인 학습**: 소스 도메인 s에 대해 다음과 같이 표현한다[1]:

$$ θ_s = K ×_0 F_s^{(0)} ×_1 F_s^{(1)} × ... ×_5 F_s^{(5)} $$

여기서:
- $$K$$: 작업 무관 코어 텐서 (task-agnostic core tensor)
- $$F_s^{(0)}, ..., F_s^{(5)}$$: 소스 도메인용 작업별 인수들 (task-specific factors)

**타겟 도메인 적응**: 새로운 타겟 도메인 t에 대해[1]:

$$ θ_t = K ×_0 F_t^{(0)} ×_1 F_t^{(1)} × ... ×_5 F_t^{(5)} $$

새로운 인수들 $$(F_t^{(0)}, ..., F_t^{(5)})$$만 학습하고 코어 텐서 $$K$$는 고정된다.

### 3. 직교성 제약 손실 함수

퇴화된 해를 방지하고 학습을 촉진하기 위해 직교성 제약을 추가한다[1]:

$$ L = λ \sum_{k=0}^{5} \|\left(F_k^{(k)}\right)^⊤F_k^{(k)} - I_d\|_F^2 $$

## 모델 구조

### 네트워크 아키텍처
**수정된 ResNet-26** 기반으로 다음과 같이 구성한다[1]:
- 3개의 매크로 모듈 (각각 64, 128, 256 채널 출력)
- 각 매크로 모듈은 4개의 기본 residual block 포함
- 각 block은 3×3 필터를 가진 두 개의 convolutional layer 포함

### 텐서 그룹화 전략
**Tucker 분해[4][5]** 개념을 확장하여 CNN의 동일한 구조 블록들을 단일 고차 텐서로 그룹화한다. 이는 기존의 4차 convolutional filter 접근법[6][7]과 달리 아키텍처 내 상관관계를 통합하는 추가 차원을 포함한다[1].

## 성능 향상

### Visual Decathlon Challenge 결과
**10개 데이터셋에서의 성능**[1]:
- **평균 정확도**: 78.43% (경쟁 방법들과 유사한 수준)
- **매개변수 효율성**: 1.35배 매개변수로 경쟁력 있는 성능 달성
- **EScore**: 2656점으로 우수한 효율성-성능 균형

### 복잡도 분석
**레이어별 Tucker 분해 대비**[1]:

$$ N_{layerwise} ≈ L × (D_0^2 + D_1^2 + D_2^2 + D_3^2) $$

**제안 방법**:

$$ N_{T-Net} = D_0^2 + D_1^2 + D_2^2 + D_3^2 + D_4^2 + D_5^2 $$

결과적으로 **L배 적은 작업별 매개변수** 사용 (ResNet-26에서 L=8)[1].

## 일반화 성능 향상 가능성

### 1. 공유된 잠재 부공간
**작업 무관 코어 텐서 K**는 모든 도메인 간에 공유되는 잠재 부공간을 나타낸다[1]. 이는 도메인 간 공통된 특징을 포착하여 새로운 도메인으로의 일반화를 촉진한다.

### 2. 저랭크 정규화 효과
**저랭크 구조**는 정규화 메커니즘으로 작용하여[8] 과적합을 방지하고 일반화 능력을 향상시킨다. 텐서의 다중선형 랭크를 제한함으로써 네트워크 전체를 효과적으로 정규화한다[1].

### 3. 제한된 데이터 환경에서의 강건성
**제한된 훈련 데이터 실험**에서 제안 방법이 전체 모델을 fine-tuning하는 강력한 기준선과 최소한 동등한 성능을 보여 모델의 강건성을 검증했다[1].

### 4. 도메인 간 지식 전이
**Inter-class Transfer Learning** 실험에서 CIFAR100으로 사전 훈련된 모델도 일부 데이터셋에서 ImageNet 모델과 유사하거나 더 나은 성능을 보여 방법의 일반화 능력을 입증했다[1].

## 한계

### 1. 아키텍처 제약
**특정 아키텍처 구조**에 의존적이며, 모든 네트워크 아키텍처에 직접 적용하기 어려울 수 있다[1].

### 2. 저랭크 제약의 한계
**ImageNet에서의 저랭크 제약** 실험에서 성능 저하가 관찰되어, 작은 ResNet 모델에서는 매개변수 수가 매우 적기 때문인 것으로 분석된다[1].

### 3. 복잡한 최적화 과정
**직교성 제약과 텐서 분해**를 동시에 최적화해야 하는 복잡성이 있으며, 하이퍼파라미터 λ의 조정이 필요하다[1].

## 앞으로의 연구에 미치는 영향

### 1. 텐서 분해 기반 신경망 압축
본 연구는 **텐서 분해를 활용한 신경망 압축**[9][10][11] 분야에 새로운 방향을 제시한다. 단순한 레이어별 압축을 넘어 구조적 그룹화를 통한 효율적인 압축 방법의 가능성을 보여준다.

### 2. 연속 학습(Continual Learning) 발전
**Catastrophic Forgetting**[12][13][14] 문제를 해결하는 새로운 접근법을 제시하여, 향후 연속 학습 연구에서 텐서 기반 방법론의 활용 가능성을 열었다.

### 3. 멀티태스크 학습 아키텍처
**Multi-Path Neural Architecture Search**[15]와 같은 다중 도메인/태스크 학습 방법론에 영감을 제공하며, 효율적인 매개변수 공유 전략의 새로운 패러다임을 제시한다.

## 향후 연구 시 고려할 점

### 1. 아키텍처 일반화
다양한 네트워크 아키텍처(Transformer, EfficientNet 등)에 적용 가능한 **일반화된 텐서 그룹화 전략** 개발이 필요하다.

### 2. 동적 랭크 조정
고정된 텐서 랭크 대신 **학습 과정에서 동적으로 조정되는 적응적 랭크 방법**[16][8] 연구가 요구된다.

### 3. 대규모 모델에의 적용
현재 연구는 상대적으로 작은 ResNet-26에서 수행되었으므로, **대규모 모델(Large Language Models, Vision Transformers)**에의 적용 가능성 탐구가 필요하다[17].

### 4. 이론적 분석 강화
**텐서 분해의 일반화 능력에 대한 이론적 보장**[18]과 수렴성 분석을 통해 방법론의 이론적 기반을 강화해야 한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/13cecd16-64a0-4736-b957-f10a2da0b343/1904.06345v2.pdf
[2] https://www.semanticscholar.org/paper/103dcba49caab4808094a336fc1a5566a7d0af0a
[3] https://cdn.aaai.org/ojs/6617/6617-13-9845-1-10-20200520.pdf
[4] https://www.jsr.org/hs/index.php/path/article/view/4916
[5] https://dl.acm.org/doi/10.1145/3409073.3409094
[6] https://linkinghub.elsevier.com/retrieve/pii/S0893608018303010
[7] https://www.semanticscholar.org/paper/23ad9b6aae4da3f519ab99f738cf7f32011bf18e
[8] https://ojs.aaai.org/index.php/AAAI/article/view/20869
[9] https://www.mdpi.com/2079-9292/11/2/214
[10] https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Towards_Efficient_Tensor_Decomposition-Based_DNN_Model_Compression_With_Optimization_Framework_CVPR_2021_paper.pdf
[11] https://ieeexplore.ieee.org/document/9892959/
[12] https://www.ibm.com/think/topics/catastrophic-forgetting
[13] https://en.wikipedia.org/wiki/Catastrophic_interference
[14] https://arxiv.org/abs/2312.10549
[15] https://research.google/blog/building-efficient-multiple-visual-domain-models-with-multi-path-neural-architecture-search/
[16] https://arxiv.org/abs/1906.07671
[17] https://arxiv.org/abs/2402.01376
[18] https://www.semanticscholar.org/paper/ac862d5a6a4352f801ba62a1f3837552b7a372b1
[19] https://linkinghub.elsevier.com/retrieve/pii/S0268401218305127
[20] https://www.ijcai.org/proceedings/2018/88
[21] https://linkinghub.elsevier.com/retrieve/pii/S0923596518310087
[22] https://www.semanticscholar.org/paper/7455328416a138a6fdd32a4a2059edd7ea7df48e
[23] https://link.springer.com/10.1007/s11030-024-10851-7
[24] https://www.esann.org/sites/default/files/proceedings/2020/ES2020-3.pdf
[25] https://openaccess.thecvf.com/content/WACV2022/papers/Garg_Multi-Domain_Incremental_Learning_for_Semantic_Segmentation_WACV_2022_paper.pdf
[26] https://openreview.net/forum?id=N7-EIciq3R
[27] https://openreview.net/forum?id=Ih9kgkeQ8k
[28] https://arxiv.org/abs/1802.04416
[29] https://velog.io/@sangwu99/TIL-Catastrophic-Forgetting-%ED%8C%8C%EA%B4%B4%EC%A0%81-%EB%A7%9D%EA%B0%81
[30] https://arxiv.org/abs/2110.12205
[31] https://velog.io/@godhj/Neural-Network-Compression-Tensor-Decomposition
[32] https://openaccess.thecvf.com/content/WACV2022/supplemental/Garg_Multi-Domain_Incremental_Learning_WACV_2022_supplemental.pdf
[33] https://wikidocs.net/253835
[34] https://www.mdpi.com/2079-9292/13/16/3279
[35] https://www.ssrn.com/abstract=4031519
[36] https://www.semanticscholar.org/paper/d2f5c589662c42b0956d2b410f8dcbfaf174bf5b
[37] https://ieeexplore.ieee.org/document/10705122/
[38] https://arxiv.org/abs/1905.09635
[39] https://sites.google.com/view/pasd/tasters
[40] https://people.lids.mit.edu/pari/nips2015.pdf
[41] https://www.tensorflow.org/datasets/catalog/visual_domain_decathlon
[42] http://papers.neurips.cc/paper/8042-statistical-mechanics-of-low-rank-tensor-decomposition.pdf
[43] https://github.com/ruihangdu/Decompose-CNN
[44] https://www.robots.ox.ac.uk/~vgg/decathlon/
[45] https://eccv.ecva.net/virtual/2024/poster/2175
[46] https://deepdata.tistory.com/1325
[47] https://www.robots.ox.ac.uk/~vgg/challenges/video-pentathlon/challenge.html
[48] https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_LowRankOcc_Tensor_Decomposition_and_Low-Rank_Recovery_for_Vision-based_3D_Semantic_CVPR_2024_paper.pdf
[49] https://openaccess.thecvf.com/content/ICCV2023/html/Huang_Video_Task_Decathlon_Unifying_Image_and_Video_Tasks_in_Autonomous_ICCV_2023_paper.html
[50] https://arxiv.org/abs/2504.13975
[51] https://ieeexplore.ieee.org/document/9723324/
[52] http://arxiv.org/pdf/1802.04416.pdf
[53] https://arxiv.org/pdf/2306.08595.pdf
[54] https://arxiv.org/pdf/2301.00314.pdf
[55] http://arxiv.org/pdf/2405.19610.pdf
[56] https://arxiv.org/pdf/2308.04595.pdf
[57] https://quantum-journal.org/papers/q-2024-06-11-1364/pdf/
[58] https://pmc.ncbi.nlm.nih.gov/articles/PMC6613218/
[59] https://arxiv.org/pdf/1501.07320.pdf
[60] https://arxiv.org/pdf/2007.07367.pdf
[61] https://arxiv.org/pdf/1907.03741.pdf
[62] https://arxiv.org/abs/1904.06345
[63] https://www.pnas.org/doi/10.1073/pnas.1611835114
[64] https://dl.acm.org/doi/10.1145/3289600.3290998
[65] https://dl.acm.org/doi/10.1609/aaai.v38i6.28359
[66] https://ieeexplore.ieee.org/document/10614384/
[67] https://ieeexplore.ieee.org/document/9460401/
[68] https://arxiv.org/pdf/1906.07671.pdf
[69] https://arxiv.org/pdf/1404.4412.pdf
[70] http://arxiv.org/pdf/2104.05758.pdf
[71] https://arxiv.org/pdf/1905.09635.pdf
[72] https://arxiv.org/pdf/2010.10131.pdf
[73] https://arxiv.org/pdf/2006.15938.pdf
[74] http://arxiv.org/pdf/2309.03439.pdf
[75] https://arxiv.org/html/2411.10218v1
[76] http://arxiv.org/pdf/2005.04366.pdf
[77] http://arxiv.org/pdf/1505.02343.pdf
[78] https://arxiv.org/abs/2204.03145
[79] https://www.sciencedirect.com/science/article/abs/pii/S0950705122000326
[80] https://paperswithcode.com/dataset/visual-domain-decathlon
