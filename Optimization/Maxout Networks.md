# Maxout Networks

### 핵심 요약

Maxout Networks는 dropout과 함께 사용하도록 설계된 새로운 활성화 함수로, 2013년 Goodfellow 등이 제안한 모델입니다. 이 논문의 핵심 주장은 dropout과 조합할 때 최적의 성능을 발휘하도록 **명시적으로 설계된 신경망 아키텍처**가 임의의 모델에 dropout을 적용하는 것보다 우수한 성능을 낼 수 있다는 것입니다. Maxout은 MNIST, CIFAR-10, CIFAR-100, SVHN 등 4개 벤치마크 데이터셋에서 당시 최고 성능을 달성했습니다.[1]

### 해결하고자 하는 문제

**1. Dropout의 근사적 모델 평균화 문제**

Dropout은 여러 하위 모델의 예측을 평균화하는 앙상블 기법으로, 단일 층 softmax 모델에서는 가중치를 2로 나누는 것만으로 정확한 모델 평균을 얻을 수 있습니다. 하지만 다층 신경망(MLP)에서는 이것이 근사치일 뿐이며, 이 근사 오차를 최소화할 필요가 있었습니다.[1]

**2. Dropout 훈련 시 최적화 문제**

Dropout 훈련은 일반적인 확률적 경사 하강법(SGD)과 다르게 작동합니다. Dropout은 매 업데이트마다 다른 하위 모델을 훈련시키는 bagging과 유사하게 작동하기 위해 **큰 학습 스텝**을 필요로 합니다. 기존 활성화 함수들은 이러한 특성에 최적화되지 않았습니다.[1]

**3. ReLU의 포화(Saturation) 문제**

Dropout과 함께 훈련할 때 ReLU 유닛은 60%가 0으로 포화되어 gradient가 차단되는 문제가 발생했습니다. 일단 0으로 포화된 뉴런은 다시 활성화되기 어려워 최적화가 어렵습니다.[1]

### 제안하는 방법: Maxout 활성화 함수

**수학적 정의**

입력 $$\mathbf{x} \in \mathbb{R}^d$$에 대해 maxout hidden layer는 다음과 같이 정의됩니다:[1]

$$h_i(\mathbf{x}) = \max_{j \in [1,k]} z_{ij}$$

여기서 $$z_{ij} = \mathbf{x}^T \mathbf{W}\_{\cdots ij} + b_{ij}$$이며, $$\mathbf{W} \in \mathbb{R}^{d \times m \times k}$$와 $$\mathbf{b} \in \mathbb{R}^{m \times k}$$는 학습 가능한 파라미터입니다.[1]

합성곱 신경망에서는 $$k$$개의 affine feature map에 대해 채널 방향으로 max pooling을 수행하여 maxout feature map을 구성합니다.[1]

**핵심 특성**

- **Piecewise Linear 근사**: 단일 maxout 유닛은 임의의 볼록 함수(convex function)를 조각별 선형 함수로 근사할 수 있습니다[1]
- **학습 가능한 활성화 함수**: Maxout 네트워크는 hidden unit 간의 관계뿐만 아니라 **각 hidden unit의 활성화 함수 자체를 학습**합니다[1]
- **항상 활성화**: ReLU와 달리 maxout은 항상 gradient가 흐르므로(locally linear almost everywhere) 포화 문제가 없습니다[1]

### 모델 구조

**Universal Approximation Theorem**

Maxout 네트워크는 **단 2개의 hidden unit**만으로도 임의의 연속 함수를 임의의 정확도로 근사할 수 있습니다(충분히 큰 $$k$$ 사용 시).[1]

이론적 근거:
1. Stone-Weierstrass 정리에 의해 임의의 연속 함수 $$f$$는 조각별 선형 함수 $$g$$로 근사 가능합니다[1]
2. $$g(v) = h_1(v) - h_2(v)$$로 표현 가능하며, 여기서 $$h_1, h_2$$는 볼록 조각별 선형 함수입니다[1]
3. 각 maxout unit은 볼록 조각별 선형 함수를 표현할 수 있으므로, 2개 unit으로 임의의 연속 함수 근사 가능합니다[1]

**실제 아키텍처**

- **MNIST (permutation invariant)**: 2개의 densely connected maxout layer + softmax layer[1]
- **MNIST (convolutional)**: 3개의 convolutional maxout layer (spatial max pooling 포함) + fully connected softmax layer[1]
- **CIFAR-10**: 3개의 convolutional maxout layer + fully connected maxout layer + fully connected softmax layer[1]

### 성능 향상

**벤치마크 결과**

| 데이터셋 | Test Error | 이전 최고 성능 |
|---------|-----------|-------------|
| MNIST (permutation invariant) | 0.94% | 0.95% (DBM)[1] |
| MNIST (convolutional) | 0.45% | 0.47% (Stochastic pooling)[1] |
| CIFAR-10 | 11.68% | 14.98% (CNN + Spearmint)[1] |
| CIFAR-10 (data augmentation) | 9.38% | 9.50% (CNN + Spearmint)[1] |
| CIFAR-100 | 38.57% | 42.51% (Stochastic pooling)[1] |
| SVHN | 2.47% | 2.68% (Rectifiers + dropout)[1] |

**Dropout과의 시너지 효과**

CIFAR-10에서 dropout 사용 시 validation error가 25% 이상 감소했습니다. Maxout은 dropout의 모델 평균화를 더 정확하게 수행합니다:[1]

- Maxout은 dropout mask 변화 시에도 **locally linear region**을 유지하여 모델 평균화가 더 정확합니다[1]
- Tanh 네트워크에 비해 maxout은 가중치를 2로 나누는 방식과 실제 geometric mean 간의 KL divergence가 더 작습니다[1]

**ReLU와의 비교**

동일한 전처리 및 하이퍼파라미터 환경에서:
- Maxout이 ReLU보다 명확한 성능 향상을 보였습니다[1]
- Cross-channel pooling 없이 동일한 필터 수를 가진 ReLU 네트워크는 maxout과 유사한 성능을 내려면 약 $$k$$배 더 많은 파라미터와 메모리가 필요합니다[1]

### 일반화 성능 향상 메커니즘

**1. 더 정확한 모델 평균화**

Dropout 훈련은 parameter sharing constraint 하에서 bagging과 유사하게 작동합니다. Maxout은 dropout mask가 변경되어도 input 주변의 **넓은 linear region**을 학습하도록 유도됩니다.[1]

실험적 검증:
- 여러 하위 모델을 샘플링하여 geometric mean을 계산한 결과, maxout이 tanh보다 가중치를 2로 나누는 근사 방법과의 일치도가 높았습니다[1]
- MNIST에서 샘플 수가 증가할수록 근사 오차가 감소하며, maxout이 더 빠르게 수렴합니다[1]

**2. Gradient Flow 개선**

Maxout은 ReLU pooling과 달리 **0을 포함하지 않아** 최적화가 개선됩니다:[1]

- ReLU pooling: 훈련 중 60%가 0으로 포화되어 gradient가 차단됩니다[1]
- Maxout: 항상 gradient가 흐르며, 0이 되더라도 파라미터의 함수이므로 조정 가능합니다[1]
- 깊은 네트워크에서 maxout은 7개 층까지 안정적으로 훈련되지만, pooled rectifier는 6-7개 층에서 성능이 크게 저하됩니다[1]

**3. 하위 층까지 변동성 전파**

MNIST 실험에서 maxout은 출력 가중치의 gradient 분산이 1.4배, **첫 번째 층 가중치의 gradient 분산이 3.4배** 더 컸습니다. 이는 dropout의 bagging 효과가 네트워크 하위 층까지 더 효과적으로 전달됨을 의미합니다.[2][1]

**4. 필터 활용도**

- ReLU with pooling: 첫 번째 층의 17.6%, 두 번째 층의 39.2% 필터가 사용되지 않았습니다[1]
- Maxout: 2400개 필터 중 2개를 제외한 모든 필터가 훈련 중 활용되었습니다[1]

### 한계점

**1. 파라미터 수 증가**

Maxout은 동일한 출력 차원에 대해 ReLU보다 $$k$$배 많은 파라미터가 필요합니다. 예를 들어, $$k=5$$인 경우 5배 많은 가중치를 학습해야 합니다.[1]

**2. 초기화 민감도**

최근 연구에 따르면, zero-mean Gaussian 분포로 초기화 시 maxout의 input-output Jacobian 분포가 **네트워크 입력에 의존**하여 불안정한 gradient와 훈련 어려움을 초래할 수 있습니다. 넓은 네트워크에서는 적절한 초기화 전략이 필수적입니다.[2]

**3. 계산 비용**

ReLU 네트워크보다 더 많은 메모리와 실행 시간이 필요합니다. 특히 $$k$$ 값이 클수록 계산 복잡도가 증가합니다.[2][1]

**4. 수렴 속도**

일부 연구에서는 maxout 네트워크가 ReLU 네트워크보다 **수렴 속도가 느리다**고 보고했습니다. 동일한 필터 수를 가진 ReLU 네트워크를 폭을 늘리는 것이 maxout을 사용하는 것보다 더 효과적일 수 있습니다.[2]

**5. 복잡도 예측 불확실성**

파라미터 공간에서 activation region의 수가 매우 다양하며, 이론적 최대값과 실제 복잡도 간에 큰 차이가 있습니다. 초기화 방법에 따라 수렴 속도가 크게 달라질 수 있습니다.[3][4]

### 향후 연구에 미치는 영향 및 고려사항

**현재까지의 영향 (2024-2025년 연구 동향 기반)**

**1. Piecewise Linear Activation Functions의 발전**

Maxout의 핵심 아이디어인 **학습 가능한 piecewise linear activation**은 지속적으로 발전하고 있습니다:

- **PWLU (Piecewise Linear Unit, 2021)**: Maxout의 확장으로, gradient-based learning으로 specialized activation function을 학습합니다. ImageNet과 COCO에서 Swish를 능가하는 성능을 보였습니다[5]
- **Adaptive activation functions (2024)**: Spline 기반의 piecewise linear function이 정확도와 모델 크기 최소화에 효과적임이 입증되었습니다[6]
- **Smoothing methods (2024)**: PLF의 미분 불가능 문제를 mollified square root function으로 해결하는 연구가 진행되었습니다[7]

**2. Dropout 정규화의 이론적 이해**

최근 연구들은 maxout의 관점에서 dropout의 작동 원리를 더 깊이 분석하고 있습니다:

- **Analytic theory of dropout (2025)**: 고차원 한계에서 dropout 동역학을 완전히 특성화하는 상미분방정식을 도출했습니다. Dropout이 hidden node 간 상관관계를 줄이고 label noise의 영향을 완화하며, 최적 dropout 확률이 데이터의 노이즈 수준에 따라 증가함을 보였습니다[8][9]
- **MC Dropout (2022)**: Monte Carlo dropout 모델이 repeatability를 크게 향상시키며, 20회 반복 후 안정화됩니다[10]

**3. 네트워크 압축 및 효율화**

Maxout의 구조적 특성을 활용한 효율화 연구:

- **Neuron pruning (2017)**: Maxout 구조를 이용한 뉴런 프루닝으로 LeNet-5는 74%, VGG16은 61% 크기 감소를 달성했습니다[11]
- **Channel-out networks (2013)**: Sparse pathway encoding 개념을 확장하여 더 효율적인 구조를 제안했습니다[12]

**4. 특수 도메인 적용**

Maxout의 적용 범위가 확대되고 있습니다:

- **의료 영역**: Protein function prediction, stress detection, posture monitoring에서 Deep Maxout Network 활용[13][14][15][16]
- **보안**: Smart city IoT 보안에서 DMN-WO 모델이 98.60% 테스트 정확도 달성[17]
- **신호 처리**: fNIRS 데이터 분류에서 대칭적 activation function의 중요성 재발견[18]

**향후 연구 시 고려사항**

**1. 초기화 전략 최적화**

- Maxout 네트워크는 ReLU처럼 잘 연구된 초기화 전략이 부족합니다[2]
- Zero-mean Gaussian 초기화 시 input-dependent gradient 문제 해결 필요[2]
- 넓고 깊은 네트워크에서 vanishing/exploding gradient 방지를 위한 새로운 초기화 방법 개발[2]

**2. 효율성과 표현력의 균형**

- 동일한 성능을 내기 위해 ReLU 네트워크의 폭을 늘리는 것과 maxout 사용 간의 tradeoff 분석 필요[2]
- $$k$$ 값 선택에 대한 체계적 가이드라인 부족: 작은 모델에서는 $$k=8$$이 $$k=4$$보다 0.8% 성능 향상[1]
- Computational overhead를 줄이면서 piecewise linear approximation의 장점을 유지하는 방법 연구[19]

**3. Batch Normalization과의 통합**

- Batch-normalized Maxout NIN (2015)은 maxout unit의 포화 문제를 완화했습니다[20]
- 최신 정규화 기법(Layer Norm, Group Norm 등)과의 조합 효과 연구 필요

**4. Transformer 및 현대 아키텍처 적용**

- Vision Transformer 검증에서 nonlinear operation으로 maxout 사용[21]
- Attention mechanism과 maxout의 결합 가능성 탐구
- Self-attention과 piecewise linear activation의 시너지 효과 분석

**5. 이론적 복잡도 분석**

- Expected complexity와 실제 성능 간 관계 정립[4][3]
- Sparse maxout networks의 표현력 이해[22]
- Neural Tangent Kernel 관점에서 maxout의 수렴 특성 분석[2]

**6. 하이브리드 접근법**

- Maxout과 다른 activation function(ReLU, Swish, GELU)의 계층별 조합[5]
- Adaptive dropout schedule과 maxout의 결합[23]
- Multi-layer maxout networks (MMN)의 추가 발전[24]

**7. 실용적 응용 확대**

- Resource-constrained device에서의 경량화 maxout 변형 개발[17]
- Real-time application을 위한 추론 속도 최적화[19]
- Few-shot learning 및 meta-learning 환경에서의 maxout 효과 검증

**결론**

Maxout Networks는 dropout과의 시너지를 통해 일반화 성능을 크게 향상시킨 혁신적인 접근법입니다. 비록 파라미터 수 증가와 계산 비용이라는 한계가 있지만, piecewise linear activation의 학습 가능성이라는 핵심 아이디어는 PWLU, 적응형 activation function, sparse pathway encoding 등으로 발전하며 현대 딥러닝 연구에 지속적인 영향을 미치고 있습니다. 향후 연구는 효율성 개선, 이론적 이해 심화, 현대 아키텍처와의 통합에 초점을 맞춰야현대 아키텍처와의 통합에 초점을 맞춰야 할 것입니다.[17][58][63][91][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d12ff649-854e-4527-9b9e-53ae9bf36eef/1302.4389v4.pdf)
[2](https://proceedings.mlr.press/v202/tseran23a/tseran23a.pdf)
[3](https://proceedings.neurips.cc/paper/2021/file/f2c3b258e9cd8ba16e18f319b3c88c66-Paper.pdf)
[4](https://papers.nips.cc/paper/2021/hash/f2c3b258e9cd8ba16e18f319b3c88c66-Abstract.html)
[5](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_Learning_Specialized_Activation_Functions_With_the_Piecewise_Linear_Unit_ICCV_2021_paper.pdf)
[6](https://arxiv.org/html/2509.18161v1)
[7](https://www.aimsciences.org/article/doi/10.3934/mfc.2023032)
[8](https://arxiv.org/html/2505.07792v1)
[9](https://journals.aps.org/pre/abstract/10.1103/jmdx-x3gr)
[10](https://www.nature.com/articles/s41746-022-00709-3)
[11](http://arxiv.org/pdf/1707.06838.pdf)
[12](http://arxiv.org/pdf/1312.1909.pdf)
[13](https://pmc.ncbi.nlm.nih.gov/articles/PMC6650051/)
[14](https://www.sciencedirect.com/science/article/abs/pii/S1746809424015003)
[15](https://www.nature.com/articles/s41598-025-04381-x)
[16](https://ieeexplore.ieee.org/document/10780953/)
[17](https://pmc.ncbi.nlm.nih.gov/articles/PMC12190705/)
[18](https://arxiv.org/html/2507.11436v1)
[19](https://dl.acm.org/doi/10.1145/3716368.3735217)
[20](http://arxiv.org/pdf/1511.02583.pdf)
[21](https://arxiv.org/abs/2405.21063)
[22](https://arxiv.org/html/2510.14068v1)
[23](https://openreview.net/forum?id=Fvfs0HPuKl)
[24](https://www.sciencedirect.com/science/article/abs/pii/S0925231217314509)
[25](http://medrxiv.org/lookup/doi/10.1101/2025.05.30.25328619)
[26](https://www.frontiersin.org/articles/10.3389/fneur.2025.1618910/full)
[27](https://www.cureus.com/articles/380972-global-research-trends-in-anticoagulation-and-mechanical-thrombectomy-for-iliofemoral-deep-vein-thrombosis-a-bibliometric-analysis-2000-2024)
[28](https://www.frontiersin.org/articles/10.3389/fonc.2025.1621666/full)
[29](https://www.frontiersin.org/articles/10.3389/fnmol.2025.1610844/full)
[30](https://jurnal.unugha.ac.id/index.php/aicp/article/view/1262)
[31](https://www.frontiersin.org/articles/10.3389/fmed.2025.1527246/full)
[32](https://jtd.amegroups.com/article/view/101282/html)
[33](https://www.mdpi.com/2076-3417/15/19/10816)
[34](https://ijcsrr.org/single-view/?id=21619&pid=21541)
[35](https://pmc.ncbi.nlm.nih.gov/articles/PMC5519034/)
[36](https://arxiv.org/pdf/2501.00779.pdf)
[37](https://arxiv.org/html/2410.04820v1)
[38](https://arxiv.org/html/2405.09357v1)
[39](https://www.linkedin.com/pulse/top-10-activation-functions-deep-learning-suresh-beekhani-vbisf)
[40](https://www.youtube.com/watch?v=Y8ve6-ifp9c)
[41](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
[42](https://journal.gnest.org/system/files/2025-02/gnest_06486_final.pdf)
[43](https://en.wikipedia.org/wiki/Activation_function)
[44](https://www.tandfonline.com/doi/abs/10.1080/19466315.2019.1697740)
[45](https://arxiv.org/pdf/2109.14545.pdf)
[46](https://www.sciencedirect.com/science/article/abs/pii/S0893608015001446)
[47](https://journal.hep.com.cn/hcc/EN/10.1016/j.hcc.2024.100238)
[48](https://hoya9802.github.io/deep-learning/maxout/)
[49](https://arxiv.org/pdf/2204.02027.pdf)
[50](https://www.sciencedirect.com/science/article/abs/pii/S0893608021000344)
[51](https://arxiv.org/abs/2411.14462)
[52](https://www.ijournalse.org/index.php/ESJ/article/view/2197)
[53](https://journal.uii.ac.id/ENTHUSIASTIC/article/view/36361)
[54](https://ejournal.seaninstitute.or.id/index.php/InfoSains/article/view/4713)
[55](https://arxiv.org/abs/2401.12362)
[56](https://mcitdoc.org.ua/index.php/ITConf/article/view/446)
[57](https://aclanthology.org/2023.semeval-1.75)
[58](https://dl.acm.org/doi/10.1145/3626641.3627213)
[59](https://ieeexplore.ieee.org/document/10714749/)
[60](https://arxiv.org/pdf/2208.04468.pdf)
[61](https://downloads.hindawi.com/journals/sp/2023/3873561.pdf)
[62](https://arxiv.org/pdf/1902.03306.pdf)
[63](http://arxiv.org/pdf/2401.10748.pdf)
[64](http://arxiv.org/pdf/2403.19896.pdf)
[65](https://arxiv.org/pdf/2302.11007.pdf)
[66](https://arxiv.org/pdf/1412.6830.pdf)
[67](https://www.emergentmind.com/topics/maxout-activation-functions)
[68](https://arxiv.org/abs/2305.06625)
[69](https://www.reddit.com/r/learnmachinelearning/comments/1mvbjwc/i_made_a_new_novel_activation_function_for_deep/)
[70](https://pubs.aip.org/aip/acp/article/3115/1/020006/3313623/Dropout-regularization-to-overcome-the-overfitting)
[71](https://arxiv.org/html/2402.00576v1)
[72](https://www.geeksforgeeks.org/deep-learning/training-neural-networks-with-dropout-for-effective-regularization/)
[73](https://www.sciencedirect.com/science/article/pii/S0957417424000472)
[74](https://www.sciencedirect.com/science/article/pii/S2773064625000143)
[75](https://www.techrxiv.org/doi/full/10.36227/techrxiv.175756440.02707962/v1)
[76](https://ieeexplore.ieee.org/iel8/6287639/10380310/10705284.pdf)
[77](https://ieeexplore.ieee.org/document/9956345/)
[78](https://ieeexplore.ieee.org/document/10903229/)
[79](https://journal.mtu.edu.iq/index.php/MTU/article/view/1420)
[80](https://www.mdpi.com/2673-6284/11/3/24)
[81](https://www.tandfonline.com/doi/full/10.1080/21681163.2022.2111720)
[82](https://ieeexplore.ieee.org/document/10315214/)
[83](https://ieeexplore.ieee.org/document/10651345/)
[84](https://journals.uic.edu/ojs/index.php/spir/article/view/12174)
[85](https://www.mdpi.com/1424-8220/23/8/3876)
[86](https://ieeexplore.ieee.org/document/10091533/)
[87](https://ieeexplore.ieee.org/document/9616637/)
[88](https://arxiv.org/pdf/1812.06169.pdf)
[89](http://arxiv.org/pdf/1701.06491.pdf)
[90](http://arxiv.org/pdf/1511.06463.pdf)
[91](https://arxiv.org/pdf/2106.04037.pdf)
[92](https://arxiv.org/pdf/2302.02508.pdf)
[93](https://www.mdpi.com/2076-3417/9/5/863/pdf?version=1551334318)
[94](http://arxiv.org/pdf/1211.3250.pdf)
[95](https://arxiv.org/pdf/2010.09458.pdf)
[96](https://businessanalyticsinstitute.com/activation-functions-impact-neural-network-performance/)
[97](https://proceedings.mlr.press/v28/goodfellow13.pdf)
[98](https://stackoverflow.com/questions/68375993/in-a-convolutional-neural-network-how-do-i-use-maxout-instead-of-relu-as-an-act)
[99](https://wikidocs.net/163752)
[100](https://proceedings.neurips.cc/paper_files/paper/2022/file/952b691c116bf753daafa6ce274e81bb-Paper-Conference.pdf)
[101](https://sites.gatech.edu/omscs7641/2024/01/31/navigating-neural-networks-exploring-state-of-the-art-activation-functions/)
[102](https://www.cscjournals.org/manuscript/Journals/IJAE/Volume1/Issue4/IJAE-26.pdf)
[103](https://www.sciencedirect.com/science/article/abs/pii/S0960077923012651)
[104](https://arxiv.org/pdf/2107.00379.pdf)
[105](https://towardsdatascience.com/activation-functions-in-neural-networks-how-to-choose-the-right-one-cb20414c04e5/)
[106](https://www.sciencedirect.com/science/article/pii/S2405896323012648)
[107](https://milvus.io/ai-quick-reference/why-are-activation-functions-important-in-neural-networks)
