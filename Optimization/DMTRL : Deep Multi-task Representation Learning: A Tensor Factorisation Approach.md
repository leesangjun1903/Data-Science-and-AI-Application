# Deep Multi-task Representation Learning: A Tensor Factorisation Approach | Domain adaption

## 핵심 주장과 주요 기여

Yang과 Hospedales가 제안한 Deep Multi-task Representation Learning (DMTRL)은 **전통적인 선형 다중 작업 학습의 한계를 극복하고 딥러닝 환경에서 자동화된 지식 공유를 실현하는 혁신적인 접근법**을 제시한다[1]. 이 연구의 핵심 기여는 다음과 같다:

**주요 기여**
- 기존 다중 작업 학습 방법들이 사용하던 행렬 분해(matrix factorization) 기법을 텐서 분해(tensor factorization)로 일반화하여 딥 네트워크의 모든 계층에서 작업 간 지식 공유 구조를 자동으로 학습[1][2]
- 사용자가 수동으로 정던 다중 작업 공유 전략을 데이터 기반으로 자동 학습하는 프레임워크 제공[1][3]
- 동질적(homogeneous) 및 이질적(heterogeneous) 다중 작업 학습 모두에 적용 가능한 범용적 접근법 개발[1]

## 해결하고자 하는 문제

### 기존 방법의 한계점
기존 딥러닝 기반 다중 작업 학습은 다음과 같은 구조적 문제를 가지고 있었다[1]:

1. **수동적 공유 전략**: 연구자가 직접 어떤 계층을 공유하고 분리할지 결정해야 함
2. **경직된 공유 방식**: 완전 공유(hard sharing) 또는 완전 분리만 가능
3. **설계 복잡성**: 작업 특화 계층 수, 작업 독립 계층 수 등을 수동 결정

### 텐서 분해의 필요성
딥 네트워크의 매개변수들은 본질적으로 텐서 구조를 가지고 있다. 완전연결 계층의 가중치는 2차원 텐서(행렬), 합성곱 계층의 커널은 3-4차원 텐서이다. 여러 작업에 대한 이러한 매개변수들을 쌓으면 자연스럽게 고차원 텐서가 형성되며, 이를 통해 지식 공유를 실현할 수 있다[1][2].

## 제안하는 방법과 수식

### 핵심 아이디어: 행렬에서 텐서로
기존 선형 다중 작업 학습에서 T개 작업의 D차원 가중치 벡터들은 D × T 행렬 W로 표현된다. 이를 다음과 같이 분해할 수 있다[1]:

**행렬 기반 지식 공유**:

$$ W = LS $$

여기서 L은 D × K 공유 행렬, S는 K × T 작업별 행렬이다. i번째 작업의 모델은:

$$ w^{(i)} = W_{:,i} = LS_{:,i} = \sum_{k=1}^{K} L_{:,k}S_{k,i} $$

### 텐서 분해 방법론

**1. Tucker 분해**
N차원 텐서 W에 대해 Tucker 분해는 다음과 같이 정의된다[1]:

$$ W = S \bullet U^{(1)} \bullet U^{(2)} \cdots \bullet U^{(N)} $$

여기서 S는 코어 텐서, $U^{(n)}$ 은 각 모드의 인수 행렬이다.

**2. Tensor Train (TT) 분해**

TT 분해는 다음과 같이 표현된다[1]:

$$ W_{d_1,d_2,...,d_N} = U^{(1)}\_{d_1,:}U^{(2)}\_{:,d_2,:}U^{(3)}\_{:,d_3,:} \cdots U^{(N)}_{:,d_N} $$

### DMTRL 프레임워크
DMTRL은 각 작업마다 동일한 구조의 DNN을 학습하되, 각 계층의 가중치를 다음 공유 구조 중 하나로 생성한다[1]:

- **DMTRL-LAF**: $$ W = LS $$ (최단순)
- **DMTRL-Tucker**: Tucker 분해 사용
- **DMTRL-TT**: Tensor Train 분해 사용

이 방법은 표준 역전파로 학습 가능하며, 순전파에서는 합성된 가중치 텐서로 추론을 수행한다[1].

## 모델 구조와 특징

### 소프트 공유 메커니즘
기존 방법들이 계층을 완전히 공유하거나 완전히 분리하는 하드 공유를 사용한 반면, DMTRL은 **소프트 공유**를 실현한다[1]. 이는 공유와 분리 사이의 연속적인 보간을 가능하게 하며, 각 계층에서 데이터 기반으로 적절한 공유 정도를 학습한다.

### 공유 강도 측정
논문에서는 학습된 공유 구조를 정량화하기 위해 공유 강도 ρ를 다음과 같이 정의한다[1]:

```math
\rho = \frac{2}{ T(T-1) } \sum_{\substack{i < j}}  \Omega(S_{:,i}, S_{:,j})
```

여기서 Ω는 코사인 유사도 측정 함수이다. ρ = 0은 완전 분리(STL), ρ = 1은 완전 공유를 의미한다.

## 성능 향상 및 일반화 성능

### 실험 결과 요약
논문에서는 세 가지 데이터셋에서 실험을 수행했다[1]:

**1. MNIST (동질적 MTL)**
- 10개 이진 분류 작업으로 구성
- DMTRL-TT와 DMTRL-Tucker가 사용자 정의 MTL 및 STL을 일관되게 상회
- 특히 훈련 데이터가 적을 때 더 큰 성능 향상 (6% 미만 오류율)

**2. AdienceFaces (이질적 MTL)**
- 성별 분류(2클래스)와 연령 그룹 분류(8클래스) 작업
- DMTRL-Tucker가 STL과 사용자 정의 MTL을 지속적으로 상회
- 부정적 전이(negative transfer) 현상 방지

**3. Omniglot (다국어 문자 인식)**
- 50개 알파벳에 대한 문자 인식 작업
- 모든 DMTRL 방법이 STL 대비 우수한 성능
- 하위 계층에서 더 많은 공유, 상위 계층에서 더 적은 공유 패턴 확인

### 일반화 성능 향상 메커니즘

**자동화된 공유 구조 학습**: DMTRL은 데이터로부터 최적의 공유 구조를 자동으로 학습하여, 수동 설계보다 더 나은 일반화 성능을 달성한다[1][3].

**소프트 공유의 이점**: 하드 공유와 달리 소프트 공유는 작업 간의 관련성 정도에 따라 공유 강도를 조절할 수 있어, 부정적 전이를 방지하면서도 유용한 지식 공유를 촉진한다[1].

**계층별 적응적 공유**: 실험 결과에 따르면, DMTRL은 하위 계층에서는 더 많이 공유하고 상위 계층에서는 더 적게 공유하는 패턴을 자동으로 학습한다. 이는 딥러닝의 계층적 특성과 일치한다[1].

## 한계점

논문에서 명시적으로 언급된 주요 한계점들은 다음과 같다:

**1. 초매개변수 설정**: 텐서 분해의 랭크 선택이 필요하며, 논문에서는 SVD 기반 초기화를 통해 이를 해결하지만 여전히 사용자 개입이 필요하다[1].

**2. 계산 복잡도**: 소프트 공유 방식은 하드 공유보다 더 많은 매개변수를 요구한다. 하지만 실험에서 UD-MTL 대비 성능 향상이 단순한 용량 증가 때문이 아님을 확인했다[1].

**3. 제한된 평가**: 상대적으로 단순한 데이터셋에서만 평가되었으며, 대규모 실제 응용에서의 확장성에 대한 검증이 부족하다.

## 후속 연구에 미치는 영향

### 직접적 영향
DMTRL은 텐서 분해를 이용한 다중 작업 학습의 새로운 패러다임을 제시했으며, 후속 연구들에 다음과 같은 영향을 미쳤다:

**텐서 기반 MTL 확장**: TRMTL (Tensor Ring Multi-Task Learning)[4], MRN (Multilinear Relationship Networks)[5][6] 등이 DMTRL의 아이디어를 확장하여 더 복잡한 텐서 분해 방법을 활용했다.

**감독 학습 텐서 분해**: MULTIPAR[7][8]는 DMTRL의 개념을 의료 데이터 분야로 확장하여 감독 학습과 텐서 분해를 결합했다.

**효율적 매개변수 공유**: 최근 연구들[9]은 DMTRL의 아이디어를 사전 훈련된 모델의 효율적 미세 조정에 활용하고 있다.

### 연구 방향 확장
- **Continual Learning**: 텐서 분해를 이용한 점진적 학습[10]
- **Domain Adaptation**: 다중 도메인 학습에서의 텐서 활용[11]
- **Neural Architecture Search**: 텐서 구조를 고려한 자동 구조 탐색[12]

## 향후 연구 시 고려사항

### 1. 확장성과 실용성
**대규모 데이터셋 적용**: 현대의 대규모 데이터셋과 모델에서의 확장성 검증이 필요하다. 특히 Transformer 기반 모델에서의 적용 가능성을 탐구해야 한다[12].

**계산 효율성 개선**: 텐서 분해의 계산 비용을 더욱 줄일 수 있는 근사 방법이나 하드웨어 최적화 기법 개발이 요구된다[13].

### 2. 이론적 기반 강화
**수렴성 분석**: DMTRL의 수렴성과 최적성에 대한 이론적 보장이 부족하다. 특히 다중 작업 환경에서의 최적화 특성에 대한 깊이 있는 분석이 필요하다.

**일반화 이론**: 왜 텐서 분해가 더 나은 일반화 성능을 보이는지에 대한 이론적 설명이 요구된다.

### 3. 적응적 구조 학습
**동적 랭크 선택**: 학습 과정에서 텐서 랭크를 동적으로 조절하는 방법 개발이 필요하다[14].

**작업 관련성 자동 탐지**: 새로운 작업이 추가될 때 기존 작업들과의 관련성을 자동으로 평가하고 적절한 공유 구조를 결정하는 메커니즘이 요구된다.

### 4. 응용 분야 확장
**다중 모달 학습**: 시각, 언어, 음성 등 다양한 모달리티를 포함하는 다중 작업 학습에서의 활용[15].

**연합 학습**: 분산 환경에서의 텐서 기반 지식 공유 방법 개발이 필요하다.

**지속 학습**: 새로운 작업을 학습할 때 기존 지식의 망각을 방지하면서도 효과적인 지식 전이를 실현하는 방법[10].

DMTRL은 다중 작업 학습 분야에서 텐서 분해의 잠재력을 처음으로 체계적으로 보여준 선구적 연구로, 향후 AI 시스템의 효율성과 일반화 성능 향상에 중요한 기여를 할 것으로 전망된다[1][16].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/179c7b98-930d-4489-988a-355f9bd0d673/1605.06391v2.pdf
[2] https://homepages.inf.ed.ac.uk/thospeda/papers/yang2017deepMTL.pdf
[3] https://www.research.ed.ac.uk/files/31878418/yang2017deepMTL.pdf
[4] https://openreview.net/forum?id=BJxmXhRcK7
[5] https://caozhangjie.github.io/files/MRN17.pdf
[6] https://ise.thss.tsinghua.edu.cn/~mlong/doc/multilinear-relationship-network-nips17.pdf
[7] https://arxiv.org/abs/2208.00993
[8] https://pmc.ncbi.nlm.nih.gov/articles/PMC11611252/
[9] https://aclanthology.org/2023.findings-acl.476
[10] https://ojs.aaai.org/index.php/AAAI/article/view/6617
[11] https://www.ijcai.org/proceedings/2019/0566.pdf
[12] https://arxiv.org/html/2302.09019v3
[13] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5172392
[14] https://proceedings.mlr.press/v225/ren23a/ren23a.pdf
[15] https://ieeexplore.ieee.org/document/10350561/
[16] https://www.themoonlight.io/en/review/optimizing-multi-task-learning-for-enhanced-performance-in-large-language-models
[17] https://www.ijcai.org/proceedings/2019/393
[18] https://www.techscience.com/iasc/v27n1/41129
[19] https://link.springer.com/10.1007/978-3-030-59728-3_43
[20] https://www.mdpi.com/1999-5903/11/1/24
[21] https://link.springer.com/10.1007/s11277-023-10235-4
[22] https://www.semanticscholar.org/paper/6aeeac671877539d1f0facb18a47941dc85584b3
[23] https://arxiv.org/abs/1605.06391
[24] https://openreview.net/forum?id=SkhU2fcll
[25] https://openreview.net/forum?id=G6yq9v8O0U&noteId=14fG1hhUaz
[26] http://proceedings.mlr.press/v80/lee18d/lee18d-supp.pdf
[27] https://www.research.ed.ac.uk/en/publications/deep-multi-task-representation-learning-a-tensor-factorisation-ap
[28] http://papers.neurips.cc/paper/5628-multitask-learning-meets-tensor-factorization-task-imputation-via-convex-optimization.pdf
[29] http://yang.ac/publications/
[30] https://scispace.com/pdf/deep-multi-task-representation-learning-a-tensor-2ctmqxf93x.pdf
[31] https://dblp.org/db/conf/iclr/iclr2017w
[32] https://ieeexplore.ieee.org/document/10447695/
[33] https://dl.acm.org/doi/10.1145/3018661.3018716
[34] https://ieeexplore.ieee.org/document/8918011/
[35] https://ieeexplore.ieee.org/document/10965303/
[36] https://ieeexplore.ieee.org/document/10096241/
[37] https://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0013161400003890
[38] https://aclanthology.org/2024.argmining-1.5.pdf
[39] https://openreview.net/forum?id=HygUBaEFDB
[40] https://arxiv.org/html/2503.21507v1
[41] https://arxiv.org/abs/2007.01126
[42] https://wikidocs.net/213665
[43] https://www.jmlr.org/papers/volume25/23-0205/23-0205.pdf
[44] https://pmc.ncbi.nlm.nih.gov/articles/PMC9748469/
[45] https://www.mdpi.com/2079-9292/13/16/3279
[46] https://aclanthology.org/2024.wmt-1.113.pdf
[47] https://sunny-archive.tistory.com/158
[48] https://www.sciencedirect.com/science/article/abs/pii/S0893608025006884
[49] https://www.sciencedirect.com/science/article/pii/S2666827023000324
[50] https://ieeexplore.ieee.org/document/8170321/
[51] https://www.semanticscholar.org/paper/674d30b387de46bc530b0c2692e936cc415691a3
[52] https://arxiv.org/pdf/2310.06124.pdf
[53] https://arxiv.org/pdf/2303.02451.pdf
[54] https://arxiv.org/abs/2302.06133
[55] https://arxiv.org/html/2501.10529v1
[56] https://arxiv.org/pdf/1802.04676.pdf
[57] https://arxiv.org/html/2405.16671v1
[58] http://arxiv.org/pdf/1805.07541.pdf
[59] https://www.aclweb.org/anthology/E17-2026.pdf
[60] https://arxiv.org/pdf/1609.07222.pdf
[61] https://github.com/wOOL/DMTRL
[62] https://github.com/safooray/tensor_factorization_mtl
[63] https://openreview.net/pdf?id=BJxmXhRcK7
[64] https://www.bohrium.com/paper-details/deep-transfer-tensor-factorization-for-multi-view-learning/867749652714226034-108597
[65] https://ijwos.com/index.php/home/article/view/2
[66] https://link.springer.com/10.1007/978-3-030-01258-8_7
[67] http://arxiv.org/pdf/2309.10357.pdf
[68] https://arxiv.org/pdf/1903.12117.pdf
[69] https://arxiv.org/pdf/2312.14472v1.pdf
[70] https://arxiv.org/pdf/2503.05126.pdf
[71] https://arxiv.org/pdf/2111.10601.pdf
[72] http://arxiv.org/pdf/2402.03557.pdf
[73] http://arxiv.org/pdf/2501.04293.pdf
[74] https://velog.io/@sksmslhy/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-An-overview-of-multi-task-learning-in-deep-neural-networks
