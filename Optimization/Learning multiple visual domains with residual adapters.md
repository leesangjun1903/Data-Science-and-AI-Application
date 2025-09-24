# Learning multiple visual domains with residual adapters | Domain adaption, residual apapters

## 핵심 주장과 주요 기여

Rebuffi et al. (2017)의 "Learning multiple visual domains with residual adapters" 논문은 **다중 도메인 시각 인식을 위한 범용 신경망 아키텍처**를 제안하여 딥러닝 분야에 중요한 기여를 하였다[1]. 이 연구는 하나의 모델로 완전히 다른 여러 시각 도메인(예: 개 품종, 교통 표지판, 숫자 인식 등)을 동시에 처리할 수 있는 혁신적인 접근법을 제시한다.

## 해결하고자 하는 문제

### 다중 도메인 학습의 주요 도전과제

전통적인 딥러닝 모델은 단일 태스크나 도메인에 특화되어 있어, 새로운 도메인에 적용할 때 **파멸적 망각(catastrophic forgetting)** 문제가 발생한다[2][3]. 이는 새로운 태스크를 학습할 때 이전에 학습한 정보를 급격히 잃어버리는 현상으로, 연속 학습(continual learning) 환경에서 심각한 문제가 된지적하는 핵심 문제들은 다음과 같다:

1. **도메인 간 차이**: 자연 이미지와 의료 이미지처럼 스타일과 내용이 모두 다른 경우
2. **파라미터 효율성**: 각 도메인마다 별도의 모델을 학습하는 것은 메모리와 계산 비용이 막대함
3. **지식 전이의 한계**: 기존 전이 학습은 원본 도메인의 성능을 희생하면서 새로운 도메인에 적응

## 제안하는 방법론

### Residual Adapter 모듈

논문의 핵심 아이디어는 **residual adapter 모듈**을 통해 기존 ResNet 아키텍처를 확장하는 것이다[1]. 이 모듈의 수학적 정의는 다음과 같다:

$$ g(x; \alpha) = x + \alpha * x $$

여기서 $$\alpha$$는 1×1 컨볼루션 필터 뱅크로, 도메인별 특화 파라미터이다[1].

### 파라미터 공유 전략

전체 네트워크는 **도메인 불가지론적(domain-agnostic) 파라미터**와 **도메인 특화(domain-specific) 파라미터**로 구분된다[1]:

- **공유 파라미터** (w₁, w₂): 전체 파라미터의 90% 이상을 차지하며 모든 도메인이 공유
- **도메인 특화 파라미터** (α): 전체 파라미터의 10% 미만으로 각 도메인별로 독립적

### Batch Normalization 적응

각 adapter 모듈에는 **도메인별 Batch Normalization** 레이어가 포함되어 있어, 도메인 간 분포 차이를 효과적으로 처리한다[1]. 이는 기존 연구에서 BN 파라미터만 조정하는 것보다 훨씬 강력한 적응 능력을 제공한다[4].

## 모델 구조

### 네트워크 아키텍처

제안된 구조는 ResNet28을 기반으로 하며, 각 residual 블록에 adapter 모듈이 추가된다[1]. 전체적인 구조는 다음과 같다:

1. **공유 특징 추출기** φₐ: 모든 도메인이 공통으로 사용
2. **도메인별 분류기** ψᵈ: 각 도메인에 특화된 선형 분류기
3. **Adapter 모듈**: residual 연결과 1×1 convolution으로 구성

### 도메인 예측 메커니즘

테스트 시 도메인 정보가 없는 경우, 별도의 경량 ResNet을 통해 도메인을 예측한다[1]. 실험에서 99.8%의 도메인 예측 정확도를 달성하여 실용성을 입증했다.

## 성능 향상 및 실험 결과

### Visual Decathlon Challenge

논문에서는 **Visual Decathlon Challenge**라는 새로운 벤치마크를 제안했다[1]. 이는 10개의 서로 다른 시각 도메인에서 동시에 잘 수행하는 모델의 능력을 평가한다:

- Aircraft, CIFAR-100, Daimler Pedestrians, Describable Textures
- German Traffic Signs, ImageNet, VGG-Flowers, Omniglot, SVHN, UCF101

### 성능 비교

| 방법 | 파라미터 수 | 평균 정확도 | Decathlon 점수 |
|------|------------|------------|----------------|
| 개별 모델 (Scratch) | 10× | 70.32% | 1625 |
| Fine-tuning | 10× | 76.51% | 2500 |
| **Residual Adapters** | **2×** | **73.88%** | **2118** |
| **Res. adapt. (최적화)** | **2×** | **76.89%** | **2621** |

residual adapter 방법은 **파라미터 수를 80% 절약**하면서도 경쟁력 있는 성능을 달성했다[1].

### 망각 방지 효과

기존 fine-tuning과 달리, residual adapter는 **원본 도메인의 성능을 완전히 보존**한다[1]. 예를 들어, ImageNet에서 새로운 도메인으로 적응할 때:

- Fine-tuning: ImageNet 성능 59.87% → 다양한 수준으로 저하
- **Residual Adapters**: ImageNet 성능 59.87% → **성능 유지**

## 일반화 성능 향상

### Universal Representation Learning

본 논문의 접근법은 **범용 표현 학습(universal representation learning)**의 토대를 마련했다[5][6]. 이는 단일 신경망이 여러 태스크와 도메인에서 효과적으로 작동할 수 있는 표현을 학습하는 것을 의미한다.

### Cross-domain 일반화

실험 결과, residual adapter로 학습된 특징은 **도메인 간 전이**에서 우수한 성능을 보였다[1]. 특히:

- 작은 데이터셋에서 큰 성능 향상 (예: Flowers 데이터셋)
- 도메인 간 공통 특징의 효과적 추출
- 저수준 및 중수준 시각 특징의 공유를 통한 일반화

### 확장성

더 큰 용량의 모델(12× 파라미터)을 사용했을 때, **3131점의 Decathlon 점수**를 달성하여 개별 모델들을 능가했다[1]. 이는 방법론의 확장 가능성을 입증한다.

## 한계점

### 도메인 유사성 의존성

residual adapter의 효과는 **도메인 간 유사성**에 크게 의존한다[1]. 완전히 다른 특성을 가진 도메인들 간에는 공유할 수 있는 특징이 제한적일 수 있다.

### 최적 파라미터 설정

각 도메인에 대해 **적절한 weight decay 값**을 찾아야 하는 문제가 있다[1]. 작은 데이터셋에는 높은 정규화가, 큰 데이터셋에는 낮은 정규화가 필요하다.

### 도메인 수 증가에 따른 복잡성

도메인 수가 증가하면 **도메인별 파라미터도 선형적으로 증가**한다. 수십 개 이상의 도메인에 대해서는 메모리 효율성이 저하될 수 있다[1].

## 후속 연구에 미치는 영향

### Adapter 기법의 확산

본 연구는 **adapter 기반 파라미터 효율적 학습**의 선구적 역할을 했다[7]. 이후 자연언어처리 분야에서도 유사한 접근법이 널리 채택되었다:

- BERT의 adapter 모듈
- Vision Transformer의 adapter 기법
- 멀티모달 학습에서의 adapter 활용

### Continual Learning 발전

논문의 아이디어는 **지속적 학습(continual learning)** 분야에 큰 영향을 미쳤다[8]. 특히:

- 작업별 파라미터와 공유 파라미터의 분리
- 선택적 파라미터 업데이트를 통한 망각 방지
- 메모리 효율적인 다중 태스크 학습

### 도메인 적응 기법 진화

**도메인 적응(domain adaptation)** 연구에서 batch normalization 기반 방법들이 발전했다[9][10]:

- Adaptive Batch Normalization (AdaBN)
- Domain-specific Batch Normalization
- Test-time adaptation 기법들

## 향후 연구 고려사항

### 자동화된 Adapter 설계

향후 연구에서는 **자동화된 adapter 구조 탐색**이 필요하다. 현재는 수동으로 설계된 1×1 convolution을 사용하지만, Neural Architecture Search (NAS)를 통해 더 효과적인 구조를 찾을 수 있을 것이다.

### 대규모 도메인 확장

**수백 개 도메인**을 다루는 대규모 시나리오에서는 새로운 접근법이 필요하다:

- 계층적 도메인 구조를 활용한 adapter 설계
- 도메인 간 유사성을 고려한 동적 파라미터 공유
- 메타러닝을 통한 새로운 도메인의 빠른 적응

### 실시간 도메인 적응

**실시간 환경**에서 새로운 도메인이 등장할 때의 적응 메커니즘 개발이 중요하다:

- 온라인 학습을 통한 점진적 adapter 업데이트
- 도메인 드리프트 감지 및 대응
- 계산 효율성을 고려한 경량화 기법

### 다중 모달리티 확장

텍스트, 이미지, 음성 등 **다중 모달리티**를 아우르는 범용 adapter 설계가 미래 연구의 핵심이 될 것이다. 이는 진정한 의미의 범용 인공지능 구현에 기여할 수 있을 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9114cedc-3a27-4af8-839a-0d7a2ca509e3/1705.08045v5.pdf
[2] https://www.nightfall.ai/ai-security-101/catastrophic-forgetting
[3] https://www.ibm.com/think/topics/catastrophic-forgetting
[4] https://ieeexplore.ieee.org/document/9879070/
[5] https://arxiv.org/abs/2204.02744
[6] https://github.com/VICO-UoE/UniversalRepresentations
[7] https://arxiv.org/abs/2312.08733
[8] https://aclanthology.org/2021.emnlp-main.590
[9] https://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf
[10] https://arxiv.org/abs/1603.04779
[11] https://www.semanticscholar.org/paper/d89ee98810039d2061ed42ee8026da49c503d16b
[12] https://www.semanticscholar.org/paper/713e5c8e5d379452c4841b0bb06c905dc01aca49
[13] https://www.semanticscholar.org/paper/9c0f5f07997b439d78956b105a792a75a0285a1c
[14] https://ieeexplore.ieee.org/document/10656717/
[15] https://arxiv.org/abs/2401.00971
[16] https://linkinghub.elsevier.com/retrieve/pii/S0094114X21003682
[17] https://paperswithcode.com/paper/a-study-of-residual-adapters-for-multi-domain
[18] https://papers.nips.cc/paper/6654-learning-multiple-visual-domains-with-residual-adapters
[19] https://aclanthology.org/2020.wmt-1.72/
[20] https://hyper.ai/kr/sota/tasks/continual-learning/benchmark/continual-learning-on-visual-domain-decathlon
[21] https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/rebuffi17learning.pdf
[22] https://www.nature.com/articles/s41597-022-01721-8
[23] https://www.catalyzex.com/s/Visual%20Domain%20Decathlon
[24] https://arxiv.org/html/2403.05175v1
[25] https://arxiv.org/abs/1705.08045
[26] https://en.wikipedia.org/wiki/Catastrophic_interference
[27] https://github.com/srebuffi/residual_adapters
[28] https://www.robots.ox.ac.uk/~vgg/decathlon/
[29] https://wikidocs.net/253835
[30] https://dl.acm.org/doi/10.5555/3294771.3294820
[31] https://iopscience.iop.org/article/10.1088/1741-2552/acfe9c
[32] https://www.semanticscholar.org/paper/cbf98ebe967e0f3f3236e7932f37013b98244e94
[33] https://www.semanticscholar.org/paper/55c4a747855c74210919c45f7899e1f79e4c97f5
[34] https://arxiv.org/abs/2205.08124
[35] https://ieeexplore.ieee.org/document/10138687/
[36] https://ieeexplore.ieee.org/document/10162211/
[37] https://arxiv.org/abs/2401.01219
[38] https://hyoeun-log.tistory.com/entry/WEEK5-Multi-Task-Learning
[39] https://www.geeksforgeeks.org/machine-learning/parameter-sharing-and-typing-in-machine-learning/
[40] https://www.v7labs.com/blog/domain-adaptation-guide
[41] https://junstar92.tistory.com/87
[42] http://www.cedar.buffalo.edu/~srihari/CSE676/7.9%20ParameterSharing.pdf
[43] https://datascience0321.tistory.com/34
[44] https://seunghan96.github.io/meta/study/study-(cs330)-(2%EA%B0%95)-Multi-Task-Learning,-Transfer-Learning-Basics/
[45] https://arxiv.org/abs/2306.09380
[46] https://www.kibme.org/resources/journal/20220819151239539.pdf
[47] https://alinlab.kaist.ac.kr/resource/AI602_Lec05_Transfer_Learning.pdf
[48] https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Deep_Multitask_Learning_with_Progressive_Parameter_Sharing_ICCV_2023_paper.pdf
[49] https://paperswithcode.com/task/domain-adaptation
[50] https://pmc.ncbi.nlm.nih.gov/articles/PMC9748469/
[51] https://avivnavon.github.io/blog/parameter-sharing-in-deep-learning/
[52] https://www.mdpi.com/2076-3417/13/23/12823
[53] https://velog.io/@riverdeer/Multi-task-Learning
[54] https://www.semanticscholar.org/paper/7aa38b85fa8cba64d6a4010543f6695dbf5f1386
[55] https://www.semanticscholar.org/paper/8f683dbe9ac52a4faef2464b99eabbbba1ab211d
[56] https://arxiv.org/abs/2409.19552
[57] https://www.nature.com/articles/s41598-023-46382-8
[58] https://academic.oup.com/bioinformatics/article/38/8/2102/6502274
[59] https://iopscience.iop.org/article/10.1088/2058-9565/ad8ef0
[60] https://www.nature.com/articles/s41467-022-30070-8
[61] https://proceedings.mlr.press/v232/li23b/li23b.pdf
[62] https://openreview.net/forum?id=HdIqOGvXOF
[63] https://github.com/sainatarajan/adabn-pytorch
[64] https://arxiv.org/html/2312.09361v1
[65] https://arxiv.org/pdf/1606.09282.pdf
[66] https://arxiv.org/html/2402.09142v1
[67] https://arxiv.org/html/2312.09486v3
[68] https://scispace.com/pdf/learning-without-forgetting-34a4300f5h.pdf
[69] https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Towards_Universal_Representation_Learning_for_Deep_Face_Recognition_CVPR_2020_paper.pdf
[70] https://cvlab.postech.ac.kr/research/MemBN/
[71] https://gbjeong96.tistory.com/40
[72] https://www.themoonlight.io/en/review/neural-thermodynamics-i-entropic-forces-in-deep-and-universal-representation-learning
[73] http://www.navisphere.net/7069/adaptive-batch-normalization-for-practical-domain-adaptation/
[74] https://ieeexplore.ieee.org/document/10022622/
[75] https://journals.lww.com/10.1097/HCO.0000000000000762
[76] https://arxiv.org/pdf/1705.08045.pdf
[77] https://arxiv.org/html/2401.00971v1
[78] https://arxiv.org/pdf/2006.00996.pdf
[79] http://arxiv.org/pdf/1602.04433.pdf
[80] https://arxiv.org/pdf/1711.07714.pdf
[81] http://arxiv.org/pdf/2410.02744.pdf
[82] http://arxiv.org/pdf/2402.08249.pdf
[83] https://arxiv.org/pdf/2311.02398.pdf
[84] https://arxiv.org/pdf/2110.09574.pdf
[85] https://aclanthology.org/2021.emnlp-main.541.pdf
[86] https://openaccess.thecvf.com/content_ICCV_2019/html/Li_Episodic_Training_for_Domain_Generalization_ICCV_2019_paper.html
[87] https://velog.io/@sangwu99/TIL-Catastrophic-Forgetting-%ED%8C%8C%EA%B4%B4%EC%A0%81-%EB%A7%9D%EA%B0%81
[88] https://paperswithcode.com/dataset/visual-domain-decathlon
[89] https://paperswithcode.com/sota/continual-learning-on-visual-domain-decathlon
[90] https://linkinghub.elsevier.com/retrieve/pii/S0168169921002015
[91] https://www.semanticscholar.org/paper/7438c34a128c13e811aca2e599028bb3760e1816
[92] https://www.aclweb.org/anthology/2020.socialnlp-1.8.pdf
[93] https://arxiv.org/pdf/2009.09139.pdf
[94] https://www.mdpi.com/2076-3417/11/3/975/pdf?version=1611627937
[95] http://arxiv.org/pdf/2010.15413.pdf
[96] https://arxiv.org/pdf/1810.10703.pdf
[97] http://arxiv.org/pdf/2009.11138.pdf
[98] https://arxiv.org/abs/2005.00944
[99] http://arxiv.org/pdf/2410.15875.pdf
[100] https://aclanthology.org/2021.eacl-main.39.pdf
[101] https://arxiv.org/pdf/2306.01839.pdf
[102] https://studyglance.in/dl/display.php?tno=13&topic=Parameter-Tying-and-Sharing
[103] https://europe.naverlabs.com/eccv-2020-domain-adaptation-tutorial/
[104] https://hchoi256.github.io/lightweight/meta-transfer-continual/
[105] https://nanunzoey.tistory.com/entry/CNN-Parameter-Sharing
[106] https://arxiv.org/abs/2404.16911
[107] https://onlinelibrary.wiley.com/doi/10.1002/adts.202200037
[108] http://arxiv.org/pdf/2204.02744.pdf
[109] http://arxiv.org/pdf/2306.10792.pdf
[110] https://arxiv.org/abs/2002.11841
[111] https://arxiv.org/abs/1803.08460
[112] https://arxiv.org/pdf/2401.03717.pdf
[113] http://arxiv.org/pdf/2203.08764.pdf
[114] http://arxiv.org/pdf/2303.12032.pdf
[115] http://arxiv.org/pdf/1910.00411.pdf
[116] https://www.pnas.org/doi/pdf/10.1073/pnas.2311805121
[117] https://openaccess.thecvf.com/content/ICCV2021W/DeepMTL/papers/Oren_In_Defense_of_the_Learning_Without_Forgetting_for_Task_Incremental_ICCVW_2021_paper.pdf
[118] https://simonezz.tistory.com/82
[119] https://velog.io/@dnr6054/TTN-A-Domain-Shift-Aware-Batch-Normalization-in-Test-Time-Adaptaion
