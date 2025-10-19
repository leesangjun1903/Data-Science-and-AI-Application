# Semi-Supervised Classification with Graph Convolutional Networks

## 1. 핵심 주장과 주요 기여

이 논문은 그래프 구조 데이터에 대한 확장 가능한 반지도 학습(semi-supervised learning) 접근법을 제시합니다. 주요 기여는 두 가지입니다. 첫째, spectral graph convolution의 1차 근사(first-order approximation)를 통해 동기화된 간단하고 안정적인 layer-wise propagation rule을 도입했습니다. 둘째, 이 그래프 기반 신경망 모델이 그래프 내 노드의 빠르고 확장 가능한 반지도 분류에 사용될 수 있음을 입증했습니다.[1]

모델은 그래프 엣지 수에 선형적으로 확장되며, 로컬 그래프 구조와 노드 특징을 모두 인코딩하는 hidden layer representation을 학습합니다. Citation network와 knowledge graph 데이터셋에 대한 실험에서 관련 방법들보다 상당한 성능 향상을 보였습니다.[1]

## 2. 문제 정의와 제안 방법

### 해결하고자 하는 문제

논문은 그래프(예: citation network) 내 노드(예: 문서)를 분류하는 문제를 다루며, 레이블은 소수의 노드에만 제공됩니다. 전통적인 graph-based semi-supervised learning은 graph Laplacian regularization을 사용하는데, 다음과 같은 형태입니다:[1]

$$
L = L_0 + \lambda L_{reg}
$$

여기서:

$$
L_{reg} = \sum_{i,j} A_{ij} \|f(X_i) - f(X_j)\|^2 = f(X)^\top \Delta f(X)
$$

이 방식은 연결된 노드들이 같은 레이블을 공유할 가능성이 높다는 가정에 의존하지만, 그래프 엣지가 단순히 노드 유사성만을 인코딩하는 것이 아니라 추가 정보를 포함할 수 있다는 점에서 모델링 용량을 제한할 수 있습니다.[1]

### 제안하는 방법

저자들은 그래프 구조를 신경망 모델 $$f(X, A)$$를 사용하여 직접 인코딩하고, 레이블이 있는 모든 노드에 대해 supervised target $$L_0$$로 학습함으로써 손실 함수에서 명시적인 graph-based regularization을 회피합니다.[1]

핵심 propagation rule은 다음과 같습니다:[1]

$$
H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)
$$

여기서:
- $$\tilde{A} = A + I_N$$: self-connection이 추가된 인접 행렬
- $$\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$$: degree matrix
- $$W^{(l)}$$: layer-specific 학습 가능한 가중치 행렬
- $$\sigma(\cdot)$$: 활성화 함수 (예: ReLU)
- $$H^{(l)} \in \mathbb{R}^{N \times D}$$: $$l$$번째 레이어의 activation 행렬 ($$H^{(0)} = X$$)

### Spectral Graph Convolution에서의 유도

이 propagation rule은 spectral convolution의 1차 근사로부터 유도됩니다. Spectral graph convolution은 신호 $$x \in \mathbb{R}^N$$과 필터 $$g_\theta$$를 Fourier domain에서 곱하는 것으로 정의됩니다:[1]

$$
g_\theta \star x = U g_\theta U^\top x
$$

여기서 $$U$$는 normalized graph Laplacian $$L = I_N - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$$의 eigenvector 행렬입니다.[1]

이를 Chebyshev polynomial로 근사하면:[1]

$$
g_{\theta'} \star x \approx \sum_{k=0}^K \theta'_k T_k(\tilde{L})x
$$

여기서 $$\tilde{L} = \frac{2}{\lambda_{max}}L - I_N$$이며, 이는 $$K$$-localized convolution으로 복잡도가 $$O(|E|)$$입니다[1].

$$K=1$$로 제한하고 $$\lambda_{max} \approx 2$$로 근사하면:[1]

$$
g_\theta \star x \approx \theta\left(I_N + D^{-\frac{1}{2}}AD^{-\frac{1}{2}}\right)x
$$

수치적 불안정성을 방지하기 위해 renormalization trick을 도입합니다:[1]

$$
I_N + D^{-\frac{1}{2}}AD^{-\frac{1}{2}} \rightarrow \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}
$$

### 모델 구조

2-layer GCN의 경우 다음과 같은 형태입니다:[1]

$$
Z = f(X, A) = \text{softmax}\left(\hat{A} \text{ReLU}\left(\hat{A}XW^{(0)}\right)W^{(1)}\right)
$$

여기서 $$\hat{A} = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$$는 전처리 단계에서 계산됩니다.[1]

손실 함수는 레이블이 있는 예제에 대한 cross-entropy error입니다:[1]

$$
L = -\sum_{l \in \mathcal{Y}_L} \sum_{f=1}^F Y_{lf} \ln Z_{lf}
$$

계산 복잡도는 $$O(|E|CHF)$$로, 그래프 엣지 수에 선형적입니다[1].

## 3. 성능 향상 및 일반화

### 실험 결과

논문은 citation network (Citeseer, Cora, Pubmed)와 knowledge graph (NELL) 데이터셋에서 실험을 수행했습니다. 결과는 다음과 같습니다:[1]

| Method | Citeseer | Cora | Pubmed | NELL |
|--------|----------|------|--------|------|
| ManiReg | 60.1 | 59.5 | 70.7 | 21.8 |
| DeepWalk | 43.2 | 67.2 | 65.3 | 58.1 |
| ICA | 69.1 | 75.1 | 73.9 | 23.1 |
| Planetoid* | 64.7 | 75.7 | 77.2 | 61.9 |
| **GCN** | **70.3** | **81.5** | **79.0** | **66.0** |

GCN은 모든 데이터셋에서 baseline 방법들을 상당한 차이로 능가했습니다. 또한 wall-clock training time에서도 효율적이었습니다 (예: Cora에서 4초 vs Planetoid의 13초).[1]

### 일반화 성능 향상 요인

**1. Layer-wise Linear Formulation의 장점**

$$K=1$$로 제한함으로써, 명시적인 Chebyshev polynomial parameterization에 의존하지 않고 여러 레이어를 쌓아 풍부한 convolutional filter function을 학습할 수 있습니다. 이는 social network, citation network, knowledge graph 등 노드 degree 분포가 매우 넓은 그래프에서 로컬 neighborhood 구조에 대한 과적합 문제를 완화할 수 있습니다. 또한 고정된 계산 예산에서 더 깊은 모델을 구축할 수 있어, 여러 도메인에서 모델링 용량을 향상시킵니다.[1]

**2. Renormalization Trick**

Renormalization trick $$\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$$는 1차 모델이나 고차 Chebyshev polynomial 기반 모델보다 더 나은 예측 성능과 효율성을 제공합니다. 실험 결과, renormalization trick을 사용한 모델이 Citeseer(70.3%), Cora(81.5%), Pubmed(79.0%)에서 최고 성능을 달성했습니다.[1]

**3. 그래프 구조와 특징의 통합 학습**

GCN은 각 레이어에서 이웃 노드로부터 특징 정보를 전파하여, 레이블 정보만을 집계하는 ICA 같은 방법보다 분류 성능을 향상시킵니다. 모델은 그래프 구조와 노드 특징을 반지도 분류에 유용한 방식으로 인코딩할 수 있습니다.[1]

**4. Dropout과 정규화**

모델은 dropout과 L2 regularization을 사용하여 과적합을 방지합니다. 예를 들어, Cora/Citeseer/Pubmed에서 dropout rate 0.5와 L2 regularization $$5 \times 10^{-4}$$를 사용했습니다.[1]

## 4. 한계

논문은 여러 한계점을 명시합니다:[1]

**메모리 요구사항**: Full-batch gradient descent를 사용하여 메모리 요구량이 데이터셋 크기에 선형적으로 증가합니다. Mini-batch stochastic gradient descent를 통해 완화할 수 있지만, $$K$$개 레이어를 가진 GCN의 경우 $$K$$차 neighborhood를 메모리에 저장해야 합니다.[1]

**방향성 엣지와 엣지 특징**: 현재 프레임워크는 엣지 특징을 자연스럽게 지원하지 않으며 무방향 그래프로 제한됩니다. NELL 실험에서는 방향성 그래프를 추가 노드를 가진 무방향 이분 그래프로 변환하여 처리했습니다.[1]

**가정의 제약**: Locality (K개 레이어에 대한 K차 이웃에 의존)와 self-connection과 이웃 노드로의 엣지가 동등한 중요도를 갖는다는 가정이 있습니다. 일부 데이터셋에서는 $$\tilde{A} = A + \lambda I_N$$ 형태로 trade-off parameter $$\lambda$$를 도입하여 gradient descent로 학습하는 것이 유익할 수 있습니다.[1]

**깊은 모델의 과적합**: 7개 레이어보다 깊은 모델에서는 residual connection 없이 학습이 어려워질 수 있으며, 각 노드의 유효 context 크기가 증가하고 파라미터 수가 증가하면서 과적합 문제가 발생할 수 있습니다.[1]

## 5. 앞으로의 연구에 미치는 영향과 고려사항

### 연구에 미친 영향

이 논문은 Graph Neural Network (GNN) 분야에 중요한 영향을 미쳤습니다. Spectral graph convolution의 1차 근사를 통해 효율적이고 확장 가능한 그래프 학습을 가능하게 했으며, 이는 후속 GNN 연구의 기초가 되었습니다. 특히 반지도 학습에서 그래프 구조를 직접 모델에 통합하는 접근법을 확립했습니다.[1]

### 앞으로 고려할 점

**1. 확장성 개선**: Mini-batch SGD와 sampling 전략을 개발하여 매우 큰 그래프에 적용할 수 있도록 메모리 효율성을 향상시켜야 합니다.[1]

**2. 방향성 및 이종 그래프**: 방향성 엣지와 엣지 특징을 자연스럽게 처리할 수 있는 확장을 개발해야 합니다.[1]

**3. 모델 깊이 최적화**: Residual connection이나 다른 기법을 사용하여 더 깊은 GCN 모델을 안정적으로 학습시키고, 과적합을 방지하면서 표현력을 높이는 연구가 필요합니다.[1]

**4. 하이퍼파라미터 자동화**: Self-connection weight $$\lambda$$와 같은 하이퍼파라미터를 gradient descent로 학습하는 접근법을 탐구해야 합니다.[1]

**5. 도메인 특화 응용**: Citation network와 knowledge graph 외에도 social network, 생물학적 네트워크, 추천 시스템 등 다양한 도메인에 적용하여 일반화 성능을 검증해야 합니다.[1]

이 논문은 그래프 학습의 효율성과 확장성을 크게 향상시켰으며, 향후 연구는 이러한 기초 위에서 더 복잡한 그래프 구조와 대규모 응용을 다루는 방향으로 발전할 것으로 기대됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ddfa8949-18c2-43e0-a9ba-f7dfbe03daa1/1609.02907v4.pdf)
