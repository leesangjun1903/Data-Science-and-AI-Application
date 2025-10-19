# Graph Attention Networks

## 1. 핵심 주장과 주요 기여

Graph Attention Networks(GAT)는 그래프 구조 데이터에서 작동하는 새로운 신경망 아키텍처로, **masked self-attention 메커니즘**을 활용하여 기존 그래프 합성곱 기반 방법들의 한계를 해결합니다. GAT의 핵심 기여는 다음과 같습니다:[1]

**노드 이웃에 대한 차별적 가중치 부여**: 각 노드가 이웃 노드들의 특징에 attention을 적용함으로써, 같은 이웃 내에서도 서로 다른 중요도를 암묵적으로 할당할 수 있습니다. 이는 행렬 역연산 같은 비용이 큰 연산 없이 가능하며, 사전에 전체 그래프 구조를 알 필요도 없습니다.[1]

**Transductive 및 Inductive 학습 모두에 적용 가능**: GAT는 학습 시 완전히 보지 못한 그래프에서도 일반화할 수 있어, inductive 학습 문제에 직접 적용 가능합니다. Cora, Citeseer, Pubmed citation 네트워크 및 protein-protein interaction(PPI) 데이터셋에서 state-of-the-art 성능을 달성했습니다.[1]

**계산 효율성**: Attention 연산이 모든 노드-이웃 쌍에 대해 병렬화 가능하며, eigendecomposition 같은 비용이 큰 행렬 연산이 필요 없습니다.[1]

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조 및 성능

### 2.1 해결하고자 하는 문제

기존 그래프 신경망 접근법들은 다음과 같은 한계를 가지고 있었습니다:[1]

**Spectral 방법의 한계**: Graph Laplacian의 eigenbasis에 의존하여 학습된 필터가 그래프 구조에 종속되므로, 특정 구조에서 학습된 모델을 다른 구조의 그래프에 직접 적용할 수 없습니다.[1]

**Non-spectral 방법의 한계**: 서로 다른 크기의 이웃에 대해 작동하면서도 CNN의 weight sharing 특성을 유지하는 연산자를 정의하는 것이 어려웠습니다. 일부 방법은 각 노드 degree마다 특정 가중치 행렬을 학습하거나, 고정된 개수의 이웃을 샘플링해야 했습니다.[1]

### 2.2 제안하는 방법 (수식 포함)

GAT는 **graph attentional layer**를 기본 구성 요소로 사용합니다.[1]

**입력과 출력**: 입력은 노드 특징 집합 $$h = \{\vec{h}_1, \vec{h}_2, \ldots, \vec{h}_N\}$$이며, 여기서 $$\vec{h}_i \in \mathbb{R}^F$$입니다. 출력은 새로운 노드 특징 $$h' = \{\vec{h}'_1, \vec{h}'_2, \ldots, \vec{h}'_N\}$$, $$\vec{h}'_i \in \mathbb{R}^{F'}$$입니다.[1]

**Attention 계수 계산**: 먼저 공유 선형 변환 $$W \in \mathbb{R}^{F' \times F}$$를 모든 노드에 적용한 후, self-attention 메커니즘 $$a : \mathbb{R}^{F'} \times \mathbb{R}^{F'} \rightarrow \mathbb{R}$$을 사용하여 attention 계수를 계산합니다:[1]

$$e_{ij} = a(W\vec{h}_i, W\vec{h}_j)$$

이 계수는 노드 $$j$$의 특징이 노드 $$i$$에 얼마나 중요한지를 나타냅니다.[1]

**Masked Attention**: 그래프 구조를 주입하기 위해, masked attention을 수행합니다. 즉, $$j \in \mathcal{N}_i$$인 노드에 대해서만 $$e_{ij}$$를 계산합니다 (여기서 $$\mathcal{N}_i$$는 노드 $$i$$의 이웃입니다).[1]

**정규화**: Softmax 함수를 사용하여 계수를 정규화합니다:[1]

$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

**구체적인 Attention 메커니즘**: 실험에서는 단일 레이어 feedforward 신경망을 사용하며, LeakyReLU 비선형성을 적용합니다 ($$\alpha = 0.2$$):[1]

$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\vec{a}^T [W\vec{h}_i \| W\vec{h}_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\vec{a}^T [W\vec{h}_i \| W\vec{h}_k]\right)\right)}$$

여기서 $$\vec{a} \in \mathbb{R}^{2F'}$$는 가중치 벡터이고, $$\|$$는 concatenation 연산입니다[1].

**출력 특징 계산**: 정규화된 attention 계수를 사용하여 특징의 선형 결합을 계산합니다 (비선형성 $$\sigma$$ 적용 후):[1]

$$\vec{h}'_i = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W\vec{h}_j\right)$$

**Multi-head Attention**: 학습 과정을 안정화하기 위해 multi-head attention을 사용합니다. $$K$$개의 독립적인 attention 메커니즘이 변환을 수행하고, 그 특징들을 concatenate합니다:[1]

$$\vec{h}'_i = \|_{k=1}^K \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha^k_{ij} W^k\vec{h}_j\right)$$

최종(예측) 레이어에서는 concatenation 대신 averaging을 사용합니다:[1]

$$\vec{h}'_i = \sigma\left(\frac{1}{K}\sum_{k=1}^K \sum_{j \in \mathcal{N}_i} \alpha^k_{ij} W^k\vec{h}_j\right)$$

### 2.3 모델 구조

**Transductive 학습**: 2-레이어 GAT 모델을 사용합니다. 첫 번째 레이어는 $$K=8$$개의 attention head로 각각 $$F'=8$$개의 특징을 계산 (총 64개 특징), ELU 비선형성을 적용합니다. 두 번째 레이어는 분류를 위한 단일 attention head로 $$C$$개의 특징 (클래스 수)을 계산하고 softmax 활성화를 적용합니다. L2 정규화 ($$\lambda = 0.0005$$)와 dropout ($$p=0.6$$)을 적용합니다. Pubmed의 경우 $$K=8$$개의 출력 attention head를 사용하고 L2 정규화를 $$\lambda=0.001$$로 강화했습니다.[1]

**Inductive 학습**: 3-레이어 GAT 모델을 사용합니다. 처음 두 레이어는 각각 $$K=4$$개의 attention head로 $$F'=256$$개의 특징을 계산 (총 1024개 특징), ELU 비선형성을 적용합니다. 최종 레이어는 $$K=6$$개의 attention head로 각각 121개의 특징을 계산하며, 이를 평균화한 후 logistic sigmoid 활성화를 적용합니다. Skip connection을 중간 attentional 레이어에 적용했습니다.[1]

### 2.4 성능 향상

**Transductive 성능** (분류 정확도):[1]
- **Cora**: 83.0 ± 0.7% (GCN 81.5% 대비 1.5% 향상)
- **Citeseer**: 72.5 ± 0.7% (GCN 70.3% 대비 2.2% 향상)
- **Pubmed**: 79.0 ± 0.3% (GCN와 동등)

**Inductive 성능** (PPI 데이터셋, micro-averaged F1 score):[1]
- **GAT**: 0.973 ± 0.002
- **GraphSAGE-LSTM** (최고 성능): 0.612
- **개선된 GraphSAGE**: 0.768
- **Const-GAT** (constant attention): 0.934 ± 0.006

GAT는 최고 GraphSAGE 결과 대비 **20.5% 향상**, Const-GAT 대비 **3.9% 향상**을 달성했습니다. 이는 전체 이웃을 관찰하고 다른 이웃에 다른 가중치를 할당하는 것의 중요성을 입증합니다.[1]

### 2.5 한계점

논문에서 언급된 한계점은 다음과 같습니다:[1]

**배치 크기 제한**: 현재 구현은 sparse matrix 연산을 활용하지만, 사용된 텐서 조작 프레임워크가 rank-2 텐서에 대해서만 sparse matrix multiplication을 지원하여 레이어의 배치 처리 능력이 제한됩니다.[1]

**GPU 성능**: 그래프 구조의 규칙성에 따라 GPU가 sparse 시나리오에서 CPU 대비 큰 성능 이점을 제공하지 못할 수 있습니다.[1]

**Receptive field 제한**: 모델의 receptive field 크기가 네트워크 깊이에 의해 상한이 정해집니다 (GCN과 유사). Skip connection 같은 기술로 깊이를 적절히 확장할 수 있습니다.[1]

**중복 계산**: 모든 그래프 엣지에 대한 병렬화가 특히 분산 방식으로 수행될 때, 이웃이 겹치는 경우 많은 중복 계산을 수반할 수 있습니다.[1]

**Edge feature 미포함**: 현재 모델은 edge feature를 통합하지 않아 노드 간 관계를 나타내는 더 다양한 문제를 다루는 데 한계가 있습니다.[1]

## 3. 일반화 성능 향상 가능성

GAT는 여러 측면에서 **뛰어난 일반화 성능**을 보입니다:

### 3.1 Inductive Learning 능력

GAT의 가장 중요한 일반화 특성은 **inductive learning에 직접 적용 가능**하다는 점입니다. Attention 메커니즘이 모든 엣지에 공유된 방식으로 적용되어, 사전에 전역 그래프 구조나 모든 노드의 특징에 접근할 필요가 없습니다. 이는 다음을 의미합니다:[1]

- 학습 중에 완전히 보지 못한 그래프에서도 모델을 평가할 수 있습니다[1]
- PPI 데이터셋에서 테스트 그래프는 학습 중에 완전히 관찰되지 않았지만, GAT는 0.973의 F1 score를 달성했습니다[1]
- 그래프가 방향성이 있어도 작동합니다 (엣지 $$j \rightarrow i$$가 없으면 $$\alpha_{ij}$$를 계산하지 않으면 됨)[1]

### 3.2 구조 독립성

**Spectral 방법의 근본적 한계 극복**: 기존 spectral 접근법에서 학습된 필터는 Laplacian eigenbasis에 의존하며, 이는 그래프 구조에 종속됩니다. 따라서 특정 구조에서 학습된 모델을 다른 구조의 그래프에 직접 적용할 수 없었습니다.[1]

GAT는 **노드 특징의 유사성**을 기반으로 attention을 계산하므로, 구조적 특성에 의존하지 않습니다. 이는 다양한 그래프 구조에 대한 일반화를 가능하게 합니다.[1]

### 3.3 가변 크기 이웃 처리

GAT는 서로 다른 degree를 가진 그래프 노드에 적용 가능하며, 이웃에 임의의 가중치를 지정합니다. 이는 다음을 의미합니다:[1]

- **고정된 이웃 크기 샘플링 불필요**: GraphSAGE는 각 노드의 고정 크기 이웃을 샘플링하여 계산 footprint를 일정하게 유지하지만, 추론 시 전체 이웃에 접근할 수 없습니다. GAT는 전체 이웃과 함께 작동하여 더 나은 일반화 성능을 제공합니다.[1]
- **이웃 순서 가정 없음**: GraphSAGE-LSTM은 이웃 간 일관된 순차적 순서를 가정하며, 이를 무작위로 정렬된 시퀀스로 보정합니다. GAT는 이러한 가정이 필요 없습니다.[1]

### 3.4 실험적 증거

**PPI 데이터셋 성능**: PPI 데이터셋은 inductive learning의 전형적인 예로, 테스트 그래프가 학습 중에 완전히 보이지 않습니다. GAT는 다음 성능 향상을 보였습니다:[1]

- GraphSAGE-LSTM (0.612) 대비 **36.1% 절대 향상**
- 전체 이웃을 사용한 개선된 GraphSAGE (0.768) 대비 **20.5% 향상**[1]
- Constant attention GAT (0.934) 대비 **3.9% 향상**[1]

이는 GAT가 완전히 새로운 그래프에 대해 강력한 일반화 능력을 가지고 있음을 입증합니다.

**Feature representation의 질적 분석**: Cora 데이터셋에서 사전 학습된 GAT 모델의 첫 번째 레이어에서 추출한 특징 표현을 t-SNE로 시각화한 결과, 2D 공간에서 명확한 클러스터링이 관찰되었으며, 이 클러스터들은 데이터셋의 7개 레이블에 해당합니다. 이는 모델의 판별력을 검증합니다.[1]

### 3.5 이론적 일반화 장점

**Weight sharing**: Attention 메커니즘이 엣지 간에 공유된 신경망 계산을 사용하므로, CNN의 weight sharing 특성을 유지하면서도 그래프의 불규칙한 구조를 처리합니다.[1]

**적응적 가중치**: 같은 이웃 내의 노드에 서로 다른 중요도를 암묵적으로 할당할 수 있어, 모델 용량이 향상되고 더 복잡한 패턴을 학습할 수 있습니다.[1]

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

**Attention 메커니즘의 그래프 도메인 확장**: GAT는 attention 메커니즘을 그래프 구조 데이터에 성공적으로 적용하여, 시퀀스 기반 태스크에서 검증된 attention의 효과를 그래프 도메인으로 확장했습니다. 이는 향후 다양한 그래프 기반 문제에 attention을 적용하는 연구의 기반이 됩니다.[1]

**Inductive Graph Learning의 새로운 표준**: GAT는 inductive learning 설정에서 탁월한 성능을 보여, 완전히 새로운 그래프에 대한 일반화가 중요한 응용 분야 (예: 약물 발견, 소셜 네트워크 분석, 추천 시스템)에서 실용적인 솔루션을 제공합니다.[1]

**모델 해석 가능성**: 학습된 attention 가중치를 분석하면 해석 가능성 측면에서 이점을 얻을 수 있습니다. 이는 기계 번역 분야에서와 같이 모델의 결정 과정을 이해하는 데 도움이 됩니다.[1]

**MoNet 프레임워크와의 연결**: GAT는 MoNet의 특정 인스턴스로 재구성될 수 있어, 그래프 CNN의 통합 이론적 프레임워크 구축에 기여합니다.[1]

### 4.2 향후 연구 시 고려할 점

**실용적 문제 해결**:
- **배치 처리 개선**: Sparse matrix 연산의 배치 처리 능력을 향상시켜 더 큰 배치 크기를 처리할 수 있도록 해야 합니다.[1]
- **분산 학습 최적화**: 그래프 엣지 간 병렬화 시 중복 계산을 줄이는 효율적인 분산 학습 방법 개발이 필요합니다.[1]

**모델 해석 가능성 심화**: Attention 메커니즘을 활용하여 모델의 해석 가능성에 대한 철저한 분석을 수행해야 합니다. 이는 도메인 지식을 활용하여 학습된 attention 패턴을 적절히 해석하는 것을 포함합니다.[1]

**그래프 분류로 확장**: 현재 GAT는 노드 분류에 초점을 맞추고 있지만, 그래프 분류로 확장하는 것은 응용 관점에서 관련성이 높습니다.[1]

**Edge Feature 통합**: 노드 간 관계를 나타내는 edge feature를 통합하면 더 다양한 문제를 다룰 수 있습니다. 이는 지식 그래프, 분자 구조 예측 등에서 중요합니다.[1]

**Receptive Field 확장**: Skip connection 같은 기술을 적절히 적용하여 네트워크 깊이를 확장하고 더 넓은 receptive field를 얻는 방법을 연구해야 합니다.[1]

**대규모 그래프 처리**: 수백만 개의 노드와 엣지를 가진 대규모 그래프에서 GAT를 효율적으로 학습하고 추론하는 방법을 개발해야 합니다. 이는 메모리 효율적인 샘플링 전략과 분산 학습 프레임워크를 필요로 합니다.

**다양한 Attention 메커니즘 탐색**: 현재 GAT는 특정 attention 메커니즘(LeakyReLU 기반 feedforward network)을 사용하지만, 다른 attention 형태 (예: scaled dot-product attention, additive attention)의 효과를 탐색할 수 있습니다.[1]

**도메인 특화 응용**: GAT를 다양한 실제 도메인(생물정보학, 소셜 네트워크, 추천 시스템, 교통 네트워크 등)에 적용하고, 각 도메인의 특성에 맞게 모델을 조정하는 연구가 필요합니다.

**이론적 분석 강화**: GAT의 표현력, 수렴성, 일반화 오차에 대한 이론적 분석을 강화하여 모델의 동작을 더 잘 이해하고 개선 방향을 찾아야 합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ae4e3da4-fc72-4b08-b448-430d9cb98edd/1710.10903v3.pdf)
