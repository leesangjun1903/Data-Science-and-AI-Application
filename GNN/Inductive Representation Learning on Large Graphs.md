# GraphSAGE : Inductive Representation Learning on Large Graphs

# 핵심 요약

**GraphSAGE**(Inductive Representation Learning on Large Graphs)는 기존의 노드 임베딩 기법들이 고정된 그래프에서만 동작하는 *transductive* 한계에서 벗어나, 노드의 특성(feature) 및 이웃 정보만으로 **미지의 노드**에 대한 임베딩을 *실시간*으로 생성할 수 있는 **inductive** 프레임워크를 제안한다. 주요 기여는 다음과 같다.

- **샘플링 및 집계(Sample and Aggregate)**: 각 노드의 K-홉 이웃을 무작위로 샘플링하고, 다양한 *aggregator* 함수를 통해 특성 벡터를 집계함으로써 노드 임베딩을 생성한다.[1]
- **유연한 Aggregator 아키텍처**:  
  - Mean aggregator (GCN 유사),  
  - LSTM aggregator,  
  - Pooling aggregator(max-pooling + MLP)  
  세 가지를 제안하여, unordered한 이웃 집합에 대해 대칭적이며 학습 가능한 집계 연산을 수행한다.[1]
- **효율성**: 고정된 배치 크기와 샘플 사이즈로 계산량을 제어하며, 트랜즈덕티브 방식 대비 테스트 시 100배 이상의 속도 향상을 달성한다.[1]
- **이론적 해석 가능성**: GraphSAGE가 WL 알고리즘을 연속 근사함을 보이고, 충분히 높은 표현 차원에서 노드의 클러스터링 계수(clustering coefficient) 등을 근사 학습할 수 있음을 증명한다.[1]

# 해결 문제 및 제안 방법

## 해결하고자 하는 문제
많은 실제 그래프는 새로운 노드가 지속적으로 유입되므로  
- 기존 임베딩 방식을 매번 재학습해야 하는 비효율성  
- 새로운 노드에 대해 임베딩이 존재하지 않는 *cold-start* 문제  
를 해결하는 **inductive** 임베딩 기법이 필요하다.

## 제안하는 방법
GraphSAGE는 노드 임베딩을 *학습된 함수*로 취급하며, 학습 시 다음 과정을 수행한다.

1. **이웃 샘플링**  
   각 노드 $$v$$에 대해 깊이 $$k$$마다 최대 $$S_k$$개의 이웃 $$\mathcal{N}_k(v)$$을 무작위로 샘플링.
2. **집계 함수**  
   $$k$$층에서 노드 $$v$$의 표현 $$h_v^{(k)}$$를  

$$
     h_{\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}_k\bigl(\{h_u^{(k-1)},\, \forall u \in \mathcal{N}_k(v)\}\bigr)
   $$  

$$
     h_v^{(k)} = \sigma\!\Bigl(W_k \,\bigl[h_v^{(k-1)} \,\|\, h_{\mathcal{N}(v)}^{(k)}\bigr]\Bigr)
   $$  
   
로 갱신하며, $$\|$$는 벡터 결합, $$\sigma$$는 ReLU 활성화 함수이다[1].

3. **비지도 및 지도 손실**  

- **비지도**: 랜덤 워크 기반 주변 노드와의 유사도 학습  

$$\mathcal{L} = -\log \sigma(z_u^\top z_v) - Q\,\mathbb{E}_{v_n\sim P_n}[\log\sigma(-z_u^\top z_{v_n})]$$  
   
- **지도**: 분류용 크로스엔트로피 손실

4. **테스트 시 임베딩 생성**  
   위의 집계 과정을 통해 학습된 파라미터만으로 새로운 노드 $$\tilde v$$의 임베딩 $$z_{\tilde v}$$를 즉시 계산.

# 모델 구조 및 수식

- **Algorithm 1 (Forward Propagation)**: K-층 반복을 통해 노드 표현을 갱신  
  (상세 알고리즘은 본문 참조).[1]
- **Aggregator 함수**:  
  - Mean: $$\mathrm{MEAN}(\{h_u\})$$  
  - LSTM: 순서 없는 집합에 랜덤 순열 적용 후 LSTM  
  - Pooling: $$\max\bigl(\mathrm{MLP}(h_u)\bigr)$$  
- **Loss (비지도)**:  

$$
    \mathcal{L} = - \sum_{(u,v)\in D} \log\sigma(z_u^\top z_v)
                  -\sum_{(u,v_n)\sim P_n} \log\sigma(-z_u^\top z_{v_n})
  $$

# 성능 향상 및 한계

- **성능**: Citation, Reddit, PPI 벤치마크에서 기존 DeepWalk, raw feature, GCN 기반 대비 F1-score 평균 39–63% 향상.[1]
- **속도**: 테스트 시 GraphSAGE는 DeepWalk 대비 100배 이상 빠름.  
- **한계**:  
  - 샘플링 편향: 균일 샘플링 대신 중요도 기반 샘플링 연구 필요  
  - 그래프 유형: Directed/multi-modal 그래프 확장 필요  
  - 층수 및 차원: $$K>2$$ 시 계산량 급증

# 일반화 성능 향상 가능성

GraphSAGE는 노드 특성의 분포와 이웃 구조를 동시에 학습하므로,  
- **미지 노드**와 **완전히 새로운 그래프**(단백질 상호작용 등)에 대한 강력한 일반화 가능  
- 이론적으로 충분한 표현력과 깊이에서 **클러스터링 계수** 등 구조적 지표를 근사 학습 가능[1]
- Pooling aggregator의 대칭성 및 MLP 표현력 덕분에 노드 식별 및 구조 정보 학습이 더 용이

# 향후 연구 및 고려 사항

- **비균일 샘플링**: 노드 중요도 기반 또는 학습 가능한 샘플러 도입  
- **그래프 확장**: directed, heterogeneous, dynamic 그래프로 확장  
- **하이퍼파라미터 최적화**: 샘플 크기 $$S_k$$, 층수 $$K$$, aggregator 구조 탐색  
- **효율적 구현**: 대규모 그래프 분산 환경 지원 및 메모리 최적화  

GraphSAGE는 **실시간 임베딩**과 **강력한 일반화 능력**을 결합하여, 진화하는 대규모 그래프에서의 노드 표현 학습에 새로운 표준이 될 전망이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a9c3cad7-7e6e-4428-9937-b562bce38312/1706.02216v4.pdf)
