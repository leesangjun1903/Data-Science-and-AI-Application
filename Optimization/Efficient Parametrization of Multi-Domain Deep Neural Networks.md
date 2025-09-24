# Efficient Parametrization of Multi-Domain Deep Neural Networks | Domain apaption

## **핵심 주장 및 기여**

이 논문은 딥 뉴럴 네트워크의 근본적 한계인 **단일 도메인 특화**를 해결하기 위해 **universal parametric families** 개념을 제안합니다[1][2]. 주요 아이디어는 모든 매개변수를 완전히 공유하는 대신, 도메인간 공통 매개변수(w)와 소수의 도메인별 매개변수(α)를 효율적으로 분리하는 것입니다[1][3].

**주요 기여사항:**
- **Parallel residual adapters** 설계로 기존 Series adapters 대비 성능 향상[1][2][3]
- **Cross-domain adapter compression** 기법으로 매개변수 효율성 극대화[1][3]
- **Visual Decathlon benchmark**성[1][4]
- 기존 fine-tuning 방식 대비 우수한 transfer learning 성능[1][2]

## **해결하고자 하는 문제**

기존 딥 네트워크는 각 도메인별로 독립적인 모델 학습이 필요하여 다음과 같은 문제점이 있었습니다[1][2]:

1. **높은 저장 비용**: 각 도메인마다 완전한 모델 저장 필요
2. **제한된 일반화**: 단일 도메인에 과도하게 특화
3. **비효율적 모델 교환**: 모바일 환경에서 모델 스와핑 오버헤드
4. **전이학습 한계**: 기존 fine-tuning 방식의 성능 제약

## **제안하는 방법론**

### **1. Residual Adapter 구조**

#### **Series Residual Adapter (기존 방식)**

$$
y = \rho(x; \alpha) = x + \text{diag}_1(\alpha) * x
$$

여기서 adapter는 기존 필터와 직렬로 연결되어:

$$
z = \rho(f * x; \alpha) = (\text{diag}_1(I + \alpha) * f) * x
$$

#### **Parallel Residual Adapter (제안 방식)**

$$
y = f * x + \text{diag}_1(\alpha) * x = (f + \text{diag}_L(\alpha)) * x
$$

**핵심 차이점**: Parallel 구조는 기존 네트워크에 **plug-and-play** 방식으로 추가 가능하며, 더 간단한 아키텍처를 제공합니다[1][3].

### **2. Cross-Domain Adapter Compression**

SVD 분해를 통한 어댑터 압축:

$$
[\alpha_1 ... \alpha_T] = U\Sigma V = [U][\bar{\Sigma}][\bar{V}_1^T | ... | \bar{V}_T^T]
$$

최종적으로:

$$
\forall t = 1, ..., T : \alpha_t \approx \beta\gamma_t^T
$$

여기서 β는 도메인간 공유되는 공통 메트릭이고, γₜ는 도메인별 특화 요소입니다[1].

### **3. 네트워크 아키텍처 설계**

ResNet-26을 기반으로 세 단계로 구분:
- **Early layers** (64 channels): 저수준 특징 추출
- **Mid layers** (128 channels): 중간 수준 특징
- **Late layers** (256 channels): 고수준 도메인별 특징

**중요 발견**: 최적 성능을 위해서는 shallow와 deep layers 모두에 adaptation이 필요하지만, 필요한 변경사항은 매우 작습니다[1][3].

## **성능 향상 및 실험 결과**

### **Visual Decathlon Benchmark 결과**
- **평균 정확도**: 78.36% (기존 77.17% 대비 1.2% 향상)
- **Decathlon 점수**: 3398 (기존 3159 대비 약 240점 향상)
- **매개변수 증가율**: 기존 네트워크 대비 단 1.5배 (fine-tuning은 10배)[1]

### **Transfer Learning 성능**
- CIFAR-100, UCF-101에서 기존 fine-tuning 방식 대비 **일관된 성능 향상**
- 특히 **제한된 데이터** 환경에서 현저한 성능 개선[1][3]
- MIT Places와 같은 **대규모 데이터셋**에서만 fine-tuning이 근소하게 우세

## **일반화 성능 향상 메커니즘**

### **1. Multi-Domain Regularization Effect**
Cross-domain compression이 다중 태스크 정규화 효과를 제공하여:
- 작은 데이터셋에서 과적합 방지
- 도메인간 지식 공유를 통한 일반화 능력 향상[1]

### **2. Parameter Sharing Strategy**
- **공통 매개변수 w**: 도메인간 공유되는 범용 특징 표현
- **도메인별 매개변수 α**: 각 도메인의 특성을 반영하는 최소한의 조정

### **3. Regularization 전략**
- **Weight decay**: 데이터셋 크기에 따른 차등 적용
- **Dropout**: 넓은 사전 훈련 네트워크에서 효과적[1]

## **모델의 한계**

1. **완전한 범용성 부족**: 여전히 도메인별 매개변수(α)가 필요
2. **대규모 데이터셋 제약**: MIT Places와 같은 매우 큰 데이터셋에서는 fine-tuning이 우세
3. **아키텍처 의존성**: ResNet 기반으로 설계되어 다른 아키텍처 적용 시 추가 연구 필요

## **미래 연구에 미치는 영향**

### **즉각적 영향**
- **Parameter-efficient fine-tuning** 연구의 새로운 방향 제시[5][6]
- **Multi-domain learning** 분야의 벤치마크 설정[4][7]
- **Adapter 기반 방법론**의 널리 확산[8]

### **장기적 파급효과**
- **LoRA(Low-Rank Adaptation)** 등 현대 어댑터 기법의 이론적 기초 제공[9]
- **Continual learning** 분야에서 catastrophic forgetting 완화 방법론 발전[10]
- **Mobile AI** 환경에서 효율적 모델 배포 전략 개발

## **향후 연구 고려사항**

1. **Transformer 아키텍처 적용**: Vision Transformer 등 최신 아키텍처로의 확장 연구 필요[6]

2. **자동 어댑터 설계**: Neural Architecture Search를 통한 최적 어댑터 구조 탐색[6]

3. **Domain-agnostic learning**: 도메인 라벨 없이 동작하는 dynamic adapter 개발[7][11]

4. **Scaling law 연구**: 더 많은 도메인과 더 큰 모델에서의 scalability 검증

5. **이론적 분석 강화**: Universal approximation 관점에서의 이론적 보장 연구[12][13]

이 논문은 딥러닝 모델의 효율적 다중 도메인 학습이라는 중요한 문제에 대해 **실용적이면서도 이론적으로 견고한 해결책**을 제시하였으며, 현재까지도 parameter-efficient transfer learning 분야의 핵심 참고 문헌으로 활용되고 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/42b5afb5-c493-44a1-b302-0158cd560c5a/1803.10082v1.pdf
[2] https://ieeexplore.ieee.org/document/8578945/
[3] https://openaccess.thecvf.com/content_cvpr_2018/papers/Rebuffi_Efficient_Parametrization_of_CVPR_2018_paper.pdf
[4] https://www.semanticscholar.org/paper/d89ee98810039d2061ed42ee8026da49c503d16b
[5] https://arxiv.org/abs/2401.00971
[6] https://arxiv.org/abs/2306.09295
[7] https://www.semanticscholar.org/paper/9c0f5f07997b439d78956b105a792a75a0285a1c
[8] https://www.emergentmind.com/topics/residual-activation-adapter
[9] https://arxiv.org/abs/2404.07919
[10] https://www.semanticscholar.org/paper/35c3d1ccdb4c2014a00ce9d9a96cdbb93516d2ba
[11] https://arxiv.org/abs/2006.00996
[12] https://en.wikipedia.org/wiki/Universal_approximation_theorem
[13] https://arxiv.org/abs/2308.10534
[14] https://link.springer.com/10.1007/978-3-030-31332-6_44
[15] https://ui.adsabs.harvard.edu/abs/arXiv:2107.11359
[16] https://arxiv.org/abs/2107.11359
[17] https://arxiv.org/abs/1803.10082
[18] https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/rebuffi17learning.pdf
[19] https://guanh01.github.io/files/2022rethinking.pdf
[20] https://dspace.lib.cranfield.ac.uk/bitstream/handle/1826/20107/Neural_network-based_parametric_system_identification-2023.pdf?sequence=1
[21] https://github.com/srebuffi/residual_adapters
[22] https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Deep_Multitask_Learning_with_Progressive_Parameter_Sharing_ICCV_2023_paper.pdf
[23] https://velog.io/@hyominsta/An-Overview-of-Multi-Task-Learning-in-Deep-Neural-Networks-%EB%85%BC%EB%AC%B8-%EA%B3%B5%EB%B6%80
[24] https://proceedings.neurips.cc/paper/2015/file/02522a2b2726fb0a03bb19f2d8d9524d-Paper.pdf
[25] https://dl.acm.org/doi/10.5555/3294771.3294820
[26] http://ieeexplore.ieee.org/document/8100117/
[27] https://linkinghub.elsevier.com/retrieve/pii/S0045782524000616
[28] https://aclanthology.org/2021.emnlp-main.541.pdf
[29] https://arxiv.org/pdf/2006.00996.pdf
[30] https://arxiv.org/html/2411.07501v3
[31] https://arxiv.org/html/2401.00971v1
[32] https://arxiv.org/pdf/1603.08029v1.pdf
[33] http://arxiv.org/pdf/1803.10082.pdf
[34] https://www.aclweb.org/anthology/D16-1093.pdf
[35] https://arxiv.org/pdf/1705.08045.pdf
[36] http://arxiv.org/pdf/2410.02744.pdf
[37] http://arxiv.org/pdf/2411.09475.pdf
[38] https://bloglunit.wordpress.com/2018/12/21/multi-domain-learning-in-deep-learning/
[39] https://www.bohrium.com/paper-details/universal-approximation-of-parametric-optimization-via-neural-networks-with-piecewise-linear-policy-approximation/900808050829426747-108551
[40] https://www.robots.ox.ac.uk/~srebuffi/papers/rebuffi18.pdf
