# A Unified Perspective on Multi-Domain and Multi-Task Learning | Domain Adaption

## **핵심 주장과 주요 기여**

이 논문은 **시맨틱 디스크립터(semantic descriptor)** 개념을 도입하여 다중 도메인 학습(MDL)과 다중 태스크 학습(MTL)을 통합하는 새로운 신경망 기반 프레임워크를 제안합니다[1][2]. 기존의 MTL/MDL 알고리즘들을 시맨틱 디스크립터를 구성하는 다양한 방식으로 해석함으로써 이들을 통합된 관점에서 바라볼 수 있게 하였습니다[2][3].

### **주요 기여**

**1. 기존 알고리즘의 통합적 해석**: RMTL, FEDA, MTFL, GO-MTL 등 기존의 MTL/MDL 알고리즘들을 제안된 프레임워크의 특수한 경우로 해석하여 통합적 관점을 제공합니다[1][3].

**2. 다중 변수 시맨틱 디스크립터 활용**: 기존 방법들이 도 원자적 개체로 가정하는 것과 달리, 구조화된 메타데이터를 활용할 수 있는 다변량 시맨틱 디스크립터를 도입했습니다[1][3].

**3. 제로샷 도메인 적응(ZSDA) 개념 도입**: 제로샷 학습의 개념을 도메인 적응으로 확장하여, 새로운 도메인에 대한 모델을 해당 도메인의 시맨틱 디스크립터만으로 생성할 수 있는 새로운 문제 설정을 제안했습니다[1][2].

## **해결하고자 하는 문제와 제안 방법**

### **문제점**
기존 MTL/MDL 방법들의 주요 한계점들을 다음과 같이 지적합니다[1]:
- 도메인과 태스크를 1-of-N 인코딩으로 표현하는 범주형 접근법의 제한성
- 구조화된 도메인/태스크 메타데이터를 효과적으로 활용하지 못하는 문제
- 다중 도메인과 다중 태스크를 동시에 다루는 프레임워크의 부재

### **제안 방법**

**핵심 모델 구조**: 
논문에서 제안하는 목적 함수는 다음과 같습니다[1]:

$$ \arg \min_{P,Q} \frac{1}{M} \sum_{i=1}^{M} \left[ \frac{1}{N_i} \sum_{j=1}^{N_i} L(\hat{y}_j^{(i)}, y_j^{(i)}) \right] $$

여기서 예측값은:

$$ \hat{y}_j^{(i)} = f_P(x_j^{(i)}) \cdot g_Q(z^{(i)}) $$

이 모델은 **양방향 신경망(two-sided neural network)** 구조로 구성됩니다[1][3]:
- **좌측**: 특징 학습 네트워크 $$f_P(x)$$ - 원본 특징 벡터에서 시작
- **우측**: 모델 구성 네트워크 $$g_Q(z)$$ - 시맨틱 디스크립터에서 시작

**간단한 구현에서는 하나의 내적 층으로 충분하며, 이 경우**:
- P는 D×K 행렬, Q는 B×K 행렬
- K는 중간 층의 유닛 수, D는 특징 벡터 길이, B는 시맨틱 디스크립터 길이
- 예측은 $$(x_j^{(i)}P)(z^{(i)}Q)'$$ 기반으로 수행됩니다[1][3]

## **모델 구조의 특징**

**분산 인코딩 vs 1-of-N 인코딩**: 
기존 방법들이 사용하는 1-of-N 인코딩 대신 분산 인코딩을 활용합니다. 예를 들어, 두 개의 범주형 변수 (A,B)가 각각 두 개의 상태를 가질 때, 네 개의 서로 다른 도메인을 분산 방식으로 인코딩하여 더 나은 정보 공유를 가능하게 합니다[1][3].

**학습 설정의 확장**:
- **다중 도메인 다중 태스크(MDMT)**: 도메인과 태스크 디스크립터를 연결하여 $$[z^{(d)}, z^{(t)}]$$ 동시 학습 수행[1]
- **제로샷 학습(ZSL)**: 새로운 시맨틱 벡터를 제시하여

```math
j^* = \arg \max_j f_P(x^*) \cdot f_Q(z_j^*)
```

로 인식 수행[1]
- **제로샷 도메인 적응(ZSDA)**: 새로운 도메인의 시맨틱 디스크립터만으로 해당 도메인에 적합한 모델 생성[1]

## **성능 향상 결과**

논문에서 제시한 실험 결과들을 살펴보면 일관된 성능 향상을 보여줍니다[1]:

**School Dataset (RMSE)**:
- 다중 도메인 학습: 9.37 (기존 최고 9.46 대비 향상)
- 제로샷 도메인 적응: 10.19 (기존 10.35 대비 향상)

**Audio Recognition (Error Rate)**:
- 다중 도메인 학습 평균: 9.77% (기존 최고 11.33% 대비 약 14% 향상)
- 제로샷 도메인 적응 평균: 19.14% (기존 24.61% 대비 약 22% 향상)

**Animal with Attributes Dataset**:
- 다중 태스크 학습: 87.66% (기존 85.34% 대비 2.32%p 향상)
- 제로샷 학습: 43.79% (DAP 방법 41.03% 대비 향상)

**Restaurant & Consumer Dataset (RMSE)**:
- 0.78 (기존 최고 1.06 대비 약 26% 향상)

## **일반화 성능 향상 가능성**

### **핵심 메커니즘**

**1. 구조화된 메타데이터 활용**: 시맨틱 디스크립터를 통해 도메인/태스크 간의 구조적 관계를 명시적으로 모델링함으로써 더 효과적한 지식 전이가 가능합니다[1][3]. 예를 들어, 학교 데이터셋에서 (학교 ID, 학년) 튜플로 도메인을 표현하면 단순한 범주형 인덱스보다 훨씬 나은 정보 공유가 가능합니다[4][5].

**2. 공유 표현 학습**: 양방향 신경망 구조를 통해 도메인/태스크 간에 공유되는 특징과 고유한 특징을 동시에 학습하여 일반화 능력을 향상시킵니다[6][7].

**3. 정규화 효과**: 다중 태스크 간의 상호작용이 자연스러운 정규화 역할을 하여 과적합을 방지하고 일반화 성능을 향상시킵니다[8][5].

### **제로샷 적응의 의미**

제로샷 도메인 적응은 **미지의 도메인에 대한 강건성**을 크게 향상시킵니다[1][9]. 이는 실제 응용에서 매우 중요한 특성으로, 모든 가능한 도메인 조합에 대해 데이터를 수집하고 모델을 훈련하는 것이 불가능한 상황에서 특히 유용합니다[10][11].

## **한계점**

논문에서 언급된 주요 한계점들은 다음과 같습니다[1]:

**1. 시맨틱 디스크립터의 가정**: 현재 프레임워크는 이산 변수로 구성된 시맨틱 디스크립터를 가정하고 있어, 연속적이거나 주기적인 변수(예: 자세, 밝기, 시간)로의 확장이 필요합니다.

**2. 디스크립터 가용성**: 시맨틱 디스크립터가 항상 관찰 가능하다고 가정하는데, 실제로는 디스크립터가 누락된 상황을 다루는 개선이 필요합니다.

**3. 스케일러빌리티**: 매우 많은 수의 도메인이나 태스크가 있을 때의 확장성에 대한 검증이 부족합니다.

## **미래 연구에 미치는 영향과 고려사항**

### **학술적 영향**

**1. 패러다임 전환**: 이 연구는 MTL과 MDL을 별개의 문제로 보던 관점에서 통합된 관점으로의 패러다임 전환을 제시했습니다[2][3]. 이후 연구들에서 이러한 통합적 접근법이 널리 채택되고 있습니다[12][13].

**2. 제로샷 도메인 적응 분야 개척**: 이 논문이 처음으로 제시한 제로샷 도메인 적응 개념은 새로운 연구 분야를 개척했으며[1][14], 이후 다양한 응용 분야에서 관련 연구가 활발히 진행되고 있습니다[10][15].

**3. 시맨틱 디스크립터 활용**: 구조화된 메타데이터를 활용하는 시맨틱 디스크립터 개념은 이후 연구들에서 널리 채택되어 다양한 형태로 발전하고 있습니다[16][17].

### **실용적 고려사항**

**1. 메타데이터 품질**: 시맨틱 디스크립터의 품질이 성능에 직접적인 영향을 미치므로, 도메인 전문가의 지식을 효과적으로 인코딩하는 방법에 대한 연구가 필요합니다.

**2. 계산 효율성**: 다중 태스크 학습은 일반적으로 계산 비용을 절감하지만[18], 복잡한 시맨틱 디스크립터를 사용할 때의 계산 오버헤드를 고려해야 합니다[19].

**3. 평가 방법론**: 제로샷 도메인 적응과 같은 새로운 문제 설정에 대한 표준화된 평가 방법론과 벤치마크 개발이 필요합니다.

**4. 실제 응용**: 이론적 프레임워크를 실제 산업 문제에 적용할 때 발생할 수 있는 도전과 해결책에 대한 연구가 지속되어야 합니다[20][21].

이 연구는 다중 태스크/도메인 학습 분야에 중요한 이론적 기반을 제공했으며[4][22], 향후 연구에서는 더욱 복잡하고 현실적인 시나리오에서의 적용 가능성을 탐구하는 것이 중요할 것으로 보입니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c6a3e194-31ee-4231-a89e-3ce7dcf95e5f/1412.7489v3.pdf
[2] https://www.semanticscholar.org/paper/b4482761879009635e04d170f9f9c70a74f4ba39
[3] https://www.research.ed.ac.uk/files/55653891/A_Unified_Perspective_on_Multi_Domain_and_Multi_Task_Learning.pdf
[4] https://ieeexplore.ieee.org/document/10929547/
[5] https://www.lyzr.ai/glossaries/multi-task-learning/
[6] https://link.springer.com/10.1007/978-3-031-61572-6_3
[7] https://ieeexplore.ieee.org/document/8170321/
[8] https://www.semanticscholar.org/paper/b7e531104c95eb67ce6c20d6dc1318e8bf837bf8
[9] https://www.ibm.com/think/topics/zero-shot-learning
[10] https://arxiv.org/abs/2108.05137
[11] https://en.wikipedia.org/wiki/Zero-shot_learning
[12] https://link.springer.com/10.1007/s00799-023-00392-z
[13] https://dl.acm.org/doi/10.1145/3529372.3530922
[14] https://openaccess.thecvf.com/content_WACV_2020/papers/Ishii_Partially_Zero-shot_Domain_Adaptation_from_Incomplete_Target_Data_with_Missing_WACV_2020_paper.pdf
[15] https://openaccess.thecvf.com/content_ECCV_2018/papers/Kuan-Chuan_Peng_Zero-Shot_Deep_Domain_ECCV_2018_paper.pdf
[16] https://arxiv.org/abs/2312.08636
[17] https://dl.acm.org/doi/10.1145/3690771.3690790
[18] https://aclanthology.org/2024.argmining-1.5.pdf
[19] https://ieeexplore.ieee.org/document/10365399/
[20] https://www.mdpi.com/2071-1050/16/2/833
[21] https://www.sciencedirect.com/science/article/abs/pii/S0950705122011625
[22] https://www.themoonlight.io/en/review/optimizing-multi-task-learning-for-enhanced-performance-in-large-language-models
[23] https://www.semanticscholar.org/paper/d7fd3dedb6b260702ed5e4b9175127815286e8da
[24] https://aclanthology.org/2024.semeval-1.152
[25] https://homepages.inf.ed.ac.uk/thospeda/papers/yang2015mtlmdl.pdf
[26] https://arxiv.org/abs/1412.7489
[27] https://ui.adsabs.harvard.edu/abs/2016arXiv161109345Y/abstract
[28] https://openreview.net/forum?id=r1fO8oC9Y7
[29] https://homepages.inf.ed.ac.uk/thospeda/downloads/TASKCV2016-Presentation.pdf
[30] https://arxiv.org/pdf/1611.09345.pdf
[31] https://paperswithcode.com/search?q=author%3AYongxin+Yang&order_by=stars
[32] https://arxiv.org/abs/2411.04760
[33] https://www.sciencedirect.com/science/article/abs/pii/S0925231217306847
[34] https://openreview.net/forum?id=G6yq9v8O0U&noteId=14fG1hhUaz
[35] https://oasis.postech.ac.kr/handle/2014.oak/117215
[36] https://www.semanticscholar.org/paper/3d9c59f8a6a5be7b77b210019ff072da41e36f9e
[37] https://link.springer.com/10.1007/s10822-023-00500-w
[38] https://www.isca-archive.org/interspeech_2023/ryu23_interspeech.html
[39] https://arxiv.org/abs/2412.06249
[40] https://stats.stackexchange.com/questions/551671/domain-generalization-vs-domain-adaptation
[41] https://arxiv.org/pdf/2103.03097.pdf
[42] https://proceedings.mlr.press/v205/niemeijer23a/niemeijer23a.pdf
[43] https://www.coursera.org/articles/what-is-zero-shot-learning
[44] https://openaccess.thecvf.com/content/WACV2024/papers/Niemeijer_Generalization_by_Adaptation_Diffusion-Based_Domain_Extension_for_Domain-Generalized_Semantic_Segmentation_WACV_2024_paper.pdf
[45] https://kubig-2022-2.tistory.com/54
[46] https://masterzone.tistory.com/77
[47] https://openaccess.thecvf.com/content_CVPR_2019/papers/Gong_DLOW_Domain_Flow_for_Adaptation_and_Generalization_CVPR_2019_paper.pdf
[48] https://cartinoe5930.tistory.com/entry/Zero-shot-One-shot-Few-shot-Learning%EC%9D%B4-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C
[49] https://arxiv.org/abs/2307.01401
[50] https://ieeexplore.ieee.org/document/9522821/
[51] https://linkinghub.elsevier.com/retrieve/pii/S1569843223001164
[52] https://www.aclweb.org/anthology/N15-1092.pdf
[53] https://aclanthology.org/2021.starsem-1.16.pdf
[54] http://arxiv.org/pdf/1706.05137.pdf
[55] https://arxiv.org/pdf/2209.07689.pdf
[56] https://www.aclweb.org/anthology/W17-2612.pdf
[57] https://arxiv.org/pdf/2105.11902.pdf
[58] https://arxiv.org/pdf/2309.16921.pdf
[59] http://arxiv.org/pdf/1906.00097.pdf
[60] https://www.themoonlight.io/ko/review/zero-shot-temporal-resolution-domain-adaptation-for-spiking-neural-networks
[61] https://github.com/WeiHongLee/Awesome-Multi-Domain-Multi-Task-Learning
[62] https://www.sciencedirect.com/science/article/pii/S0893608023001697
[63] https://www.mdpi.com/1099-4300/26/8/664
[64] https://www.isca-archive.org/interspeech_2023/khandelwal23_interspeech.html
[65] https://ieeexplore.ieee.org/document/9658165/
[66] https://arxiv.org/pdf/2402.16848.pdf
[67] https://arxiv.org/html/2410.05448v2
[68] http://arxiv.org/pdf/2502.11986.pdf
[69] https://arxiv.org/pdf/2312.08636.pdf
[70] http://arxiv.org/pdf/2308.12029.pdf
[71] https://arxiv.org/pdf/2110.13076.pdf
[72] https://arxiv.org/abs/2005.00944
[73] https://arxiv.org/pdf/2109.04617.pdf
[74] https://arxiv.org/html/2408.17214v1
[75] https://pmc.ncbi.nlm.nih.gov/articles/PMC6220332/
[76] https://analytics4everything.tistory.com/295
[77] https://arxiv.org/abs/2102.03137
[78] https://www.sciencedirect.com/science/article/pii/S1361841524003049
[79] http://dmqa.korea.ac.kr/activity/seminar/415
