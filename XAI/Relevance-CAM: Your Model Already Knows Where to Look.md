# Relevance-CAM: Your Model Already Knows Where to Look

## 핵심 주장과 주요 기여

**Relevance-CAM**은 기존 gradient 기반 CAM 방법들의 한계를 극복하기 위해 **Layer-wise Relevance Propagation (LRP)**을 활용한 새로운 Class Activation Mapping 방법입니다. 본 논문의 핵심 주장은 **중간 레이어(intermediate layers)에서도 클래스별 특성을 효과적으로 추출**할 수 있으며, 얕은 레이어에서조차 클래스 특정 정보가 존재한다는 것입니다.[1][2][3]

### 주요 기여사항

1. **Shattered Gradient Problem 해결**: 기존 gradient 기반 CAM의 노이즈와 불연속성 문제를 LRP를 통해 해결[3][1]
2. **전 레이어 분석 가능**: 마지막 convolutional layer뿐만 아니라 중간 및 얕은 레이어에서도 의미 있는 시각화 제공[2][1]
3. **우수한 국소화 성능**: Average Drop, Average Increase, IoU 메트릭에서 기존 방법들 대비 성능 향상[1]
4. **클래스 민감도 향상**: 얕은 레이어에서도 클래스별 특성 추출 능력 입증[2][1]

## 해결하고자 하는 문제

### 1. Shattered Gradient Problem
기존 Grad-CAM과 Grad-CAM++ 방법들은 **gradient의 노이즈와 불연속성** 문제를 겪습니다. 깊은 네트워크에서 gradient가 ReLU나 Sigmoid 같은 활성화 함수를 통과하면서 포화되거나 소실되어 중간 레이어에서 부정확한 가중치를 생성합니다.[4][3][1]

### 2. False Confidence 문제
Grad-CAM은 활성화 맵의 **민감도(sensitivity)**를 측정하지만, 실제로 필요한 것은 **기여도(contribution)**입니다. 이는 모델의 결정 과정을 정확히 반영하지 못하는 문제를 야기합니다.[4][1]

### 3. 중간 레이어 분석의 한계
기존 방법들은 주로 마지막 convolutional layer에만 집중하여, 중간 레이어의 특성 추출 능력을 분석하지 못했습니다.[1][2]

## 제안하는 방법 (수식 포함)

### Relevance-CAM 수식

Relevance-CAM은 다음과 같이 정의됩니다:

$$
L^{(c,i)}_{\text{Relevance-CAM}} = \sum_k \alpha^{(c,i)}_k A^c_k
$$

여기서 가중치 $$\alpha^{(c,i)}_k$$는 relevance map의 global average pooling으로 계산됩니다:

$$
\alpha^{(c,i)}_k = \sum_{x,y} R^{(c,i)}_k(x,y)
$$

### Layer-wise Relevance Propagation (LRP)

LRP의 기본 전파 규칙인 z-rule은 다음과 같습니다:[5][1]

$$
R_i = \sum_j \frac{z^+_{ij}}{\sum_i z^+_{ij}} R_j
$$

여기서:

$$
z^+_{ij} = x_i w^+_{ij}
$$

### Contrastive Layer-wise Relevance Propagation (CLRP)

클래스 민감도를 향상시키기 위해 CLRP를 적용합니다:[2][1]

$$
R^{(L)}_n = \begin{cases}
z^{(L)}_t & \text{if } n = t \\
-\frac{z^{(L)}_t}{N-1} & \text{otherwise}
\end{cases}
$$

## 모델 구조

Relevance-CAM의 파이프라인은 다음과 같습니다:[1]

1. **Forward Propagation**: 활성화 맵 $$A^c_k$$ 추출
2. **Relevance Propagation**: LRP를 통한 relevance map $$R^{(c,i)}$$ 계산
3. **Global Average Pooling**: relevance map에서 가중치 구성 요소 획득
4. **Weighted Linear Summation**: 최종 Relevance-CAM 생성

이 과정은 **단 한 번의 forward propagation과 backward propagation**으로 완료되어 효율적입니다.[6][1]

## 성능 향상

### 정량적 평가 결과

**ResNet-50에서의 Average Drop (A.D.) 및 Average Increase (A.I.) 결과**:[1]

| Method | Layer 2 A.D. | Layer 2 A.I. | Layer 4 A.D. | Layer 4 A.I. |
|--------|--------------|--------------|--------------|--------------|
| Grad-CAM | 74.91 | 4.45 | 23.13 | 24.05 |
| Grad-CAM++ | 71.15 | 4.85 | 22.03 | 25.35 |
| Score-CAM | 56.59 | 8.8 | 21.89 | 24.65 |
| **Relevance-CAM** | **39.02** | **16.6** | **21.53** | **25.7** |

**IoU 평가 결과**:[1]

| Method | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
|--------|---------|---------|---------|---------|
| Grad-CAM | 0.12 | 0.18 | 0.22 | 0.34 |
| Grad-CAM++ | 0.13 | 0.19 | 0.22 | 0.34 |
| Score-CAM | 0.21 | 0.25 | 0.28 | 0.34 |
| **Relevance-CAM** | **0.30** | **0.32** | **0.32** | **0.34** |

### 계산 효율성

Score-CAM 대비 **500배 이상 빠른 처리 속도**를 보여줍니다:[6]
- Score-CAM: 16.67초
- Relevance-CAM: 0.031초

## 한계점

### 1. 의존성 문제
Relevance-CAM은 **LRP의 품질에 의존적**입니다. LRP의 전파 규칙 선택이나 하이퍼파라미터 설정에 따라 결과가 달라질 수 있습니다.[7][5]

### 2. 해석의 복잡성
LRP 자체가 복잡한 방법론이므로, **일반 사용자들이 결과를 해석하기 어려울 수 있습니다**. 특히 의료 영상과 같은 도메인에서는 전문가의 도메인 지식이 필요합니다.[8][4]

### 3. 계산 복잡도
Score-CAM보다는 빠르지만, **여전히 backward propagation이 필요**하여 단순한 forward pass만 필요한 방법들보다는 계산 비용이 높습니다.[4]

### 4. 특징 간 상관관계
**특징들 간의 상관관계가 높은 경우** (예: 인간 동작에서의 키네매틱 체인), 개별 특징의 중요도가 희석될 수 있습니다.[4]

## 일반화 성능 향상 가능성

### 전이 학습에서의 활용
Relevance-CAM을 통한 **레이어별 분석**은 전이 학습에서 어떤 레이어를 fine-tuning할지 결정하는 데 도움을 줄 수 있습니다. 기존의 경험적 접근법 대신 **데이터 기반의 레이어 선택**이 가능합니다.[1]

### 모델 프루닝 최적화
중간 레이어에서의 **클래스별 특성 추출 능력**을 분석하여, 불필요한 레이어나 채널을 제거하는 **모델 압축에 활용**할 수 있습니다.[1]

### 약한 지도 학습 향상
**높은 해상도의 히트맵 생성**은 약한 지도 학습 기반 분할(weakly supervised segmentation) 성능을 향상시킬 수 있습니다. 특히 픽셀 단위 라벨이 부족한 의료 영상 분야에서 유용합니다.[9][1]

### 도메인 적응
얕은 레이어에서의 **클래스 특정 정보 추출**은 도메인 간 특성 차이를 이해하는 데 도움을 주어, 더 효과적인 도메인 적응 전략을 수립할 수 있습니다.

## 미래 연구에 미치는 영향

### 1. 새로운 해석 가능성 패러다임
Relevance-CAM은 **"모델이 이미 어디를 봐야 할지 알고 있다"**는 관점을 제시하여, 향후 해석 가능성 연구의 새로운 방향을 제시합니다.[3][1]

### 2. 멀티 레이어 분석 표준화
중간 레이어 분석의 중요성을 입증함으로써, **전체 네트워크에 대한 포괄적 분석**이 표준이 될 가능성을 높였습니다.[1]

### 3. 의료 AI의 신뢰성 향상
**픽셀 수준의 정확한 국소화**는 의료 영상 진단 AI의 신뢰성과 채택률을 높이는 데 기여할 것입니다.[8][9]

## 앞으로 연구 시 고려사항

### 1. 하이브리드 접근법 개발
SHAP와 GradCAM을 결합한 연구처럼, **Relevance-CAM과 다른 방법들을 결합**한 하이브리드 접근법 연구가 필요합니다. 각 방법의 장점을 살려 "why"와 "where"를 모두 제공하는 방향으로 발전해야 합니다.[4]

### 2. 실시간 응용을 위한 최적화
계산 효율성을 더욱 향상시켜 **실시간 진단 도구**에서 활용할 수 있도록 하는 연구가 필요합니다.[4]

### 3. 도메인별 특화 연구
의료 영상, 자율주행, 산업 검사 등 **특정 도메인에 특화된 Relevance-CAM 변형**을 개발하여 각 분야의 요구사항을 충족해야 합니다.[10][8]

### 4. 사용자 중심 설계
**다양한 이해관계자의 요구**에 맞는 설명 방식을 제공하는 연구가 필요합니다. 의사, 환자, 연구자별로 다른 수준의 설명을 제공할 수 있는 적응형 시스템 개발이 중요합니다.[4]

### 5. 검증 및 평가 기준 개선
현재의 IoU, Average Drop 등의 메트릭 외에도 **도메인별 특성을 반영한 새로운 평가 기준** 개발이 필요합니다.[11][12]

[1](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Relevance-CAM_Your_Model_Already_Knows_Where_To_Look_CVPR_2021_paper.pdf)
[2](https://velog.io/@ddsqe1/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Relevance-CAM-Your-Model-Already-Knows-Where-to-Look-z4eb9ish)
[3](https://openaccess.thecvf.com/content/CVPR2021/html/Lee_Relevance-CAM_Your_Model_Already_Knows_Where_To_Look_CVPR_2021_paper.html)
[4](https://arxiv.org/html/2412.16003v1)
[5](https://pure.korea.ac.kr/en/publications/layer-wise-relevance-propagation-an-overview)
[6](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Lee_Relevance-CAM_Your_Model_CVPR_2021_supplemental.pdf)
[7](https://iphome.hhi.de/samek/pdf/MonXAI19.pdf)
[8](https://link.springer.com/10.1007/978-3-031-58181-6_11)
[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC11178611/)
[10](https://arxiv.org/abs/2506.22866)
[11](https://arxiv.org/abs/2104.10252)
[12](https://www.nature.com/articles/s41598-025-14060-6)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/d28762db-2a40-4f4e-a65f-272dae441133/Lee_Relevance-CAM_Your_Model_Already_Knows_Where_To_Look_CVPR_2021_paper.pdf)
[14](https://www.semanticscholar.org/paper/c95217634366a11cd9efff58b55a9d9dc6e59f4d)
[15](http://journals.rudn.ru/semiotics-semantics/article/view/27566)
[16](https://et.iphras.ru/article/view/6968)
[17](https://www.mdpi.com/1099-4300/23/11/1375)
[18](https://www.jstage.jst.go.jp/article/bss/35/0/35_1/_article)
[19](https://ieeexplore.ieee.org/document/9523049/)
[20](https://www.aclweb.org/anthology/11.textgraphs-1.17)
[21](https://abjournals.org/ajsshr/papers/volume-4/issue-3/the-relevance-of-community-involvement-in-military-counter-insurgency-operations-in-north-eastern-nigeria/)
[22](https://vcot.info/magazine)
[23](https://www.ssrn.com/abstract=3837386)
[24](https://arxiv.org/html/2405.12175v1)
[25](https://arxiv.org/html/2412.05686v1)
[26](https://www.sciencedirect.com/science/article/pii/S0306261923014435)
[27](https://www.sciencedirect.com/science/article/abs/pii/S0031320324007076)
[28](https://arxiv.org/html/2506.01636v1)
[29](https://www.themoonlight.io/en/review/neural-network-interpretability-with-layer-wise-relevance-propagation-novel-techniques-for-neuron-selection-and-visualization)
[30](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321385/)
[31](https://yassouali.github.io/ml-blog/cvpr2021/)
[32](https://diglib.eg.org/server/api/core/bitstreams/993677d2-377d-4cba-8a8b-369d6c8d22b3/content)
[33](https://www.paperdigest.org/2021/06/cvpr-2021-highlights/)
[34](https://towardsdatascience.com/visualizing-neural-networks-decision-making-process-part-2-layer-wise-relevance-propagation-50cd913cc1c7/)
[35](https://openaccess.thecvf.com/CVPR2021?day=all)
[36](https://pmc.ncbi.nlm.nih.gov/articles/PMC6685087/)
[37](https://towardsdatascience.com/10-promising-synthesis-papers-from-cvpr-2021-5dc670981872/)
[38](https://najfnr.com/home/article/view/605)
[39](https://journals.lww.com/10.4103/ijcm.ijcm_535_23)
[40](https://ieeexplore.ieee.org/document/11101326/)
[41](https://lifescienceglobal.com/pms/index.php/ijsmr/article/view/9043)
[42](https://wires.onlinelibrary.wiley.com/doi/10.1002/wnan.1944)
[43](https://dl.acm.org/doi/10.1145/3477495.3531890)
[44](https://www.mdpi.com/1999-4923/14/2/402)
[45](https://www.mdpi.com/2072-6694/11/3/295)
[46](https://pmc.ncbi.nlm.nih.gov/articles/PMC9925800/)
[47](https://arxiv.org/html/2502.14416)
[48](https://arxiv.org/pdf/1912.08142.pdf)
[49](https://arxiv.org/pdf/2307.10506.pdf)
[50](https://arxiv.org/html/2301.01060v2)
[51](https://angeloyeo.github.io/2019/08/17/Layerwise_Relevance_Propagation.html)
[52](https://www.sciencedirect.com/science/article/abs/pii/S1077314225001067)
[53](https://velog.io/@dldmstkd/Relevance-CAM-Your-Model-Already-Knows-Where-to-Look)
[54](https://arxiv.org/abs/1604.00825)
[55](https://www.sciencedirect.com/science/article/abs/pii/S0925231223002060)
[56](https://arxiv.org/html/2506.22866v1)
[57](https://velog.io/@sjinu/%EA%B0%9C%EB%85%90%EC%A0%95%EB%A6%AC-LRPLayer-wise-Relevance-Propagation)
[58](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Finer-CAM_Spotting_the_Difference_Reveals_Finer_Details_for_Visual_Explanation_CVPR_2025_paper.pdf)
[59](https://kyujinpy.tistory.com/61)
[60](https://hellopotatoworld.tistory.com/17)
[61](https://www.koreascience.kr/article/JAKO202514739605327.pdf)
[62](https://velog.io/@jus6886/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0Layer-Wise-Relevance-PropagationAn-Overview)
