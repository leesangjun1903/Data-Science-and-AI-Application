# Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection

# 핵심 요약 및 주요 기여

**주요 주장:** 그래프 신경망(Graph Neural Network, GNN)을 사기 탐지에 적용할 때, 이웃 노드 간 맥락(context), 특징(feature), 관계(relation)의 불일치(inconsistency)가 모델 성능을 저해하므로, 이를 완화하기 위한 새로운 프레임워크 **GraphConsis**를 제안한다.[1]

**주요 기여:**  
- 사기 탐지용 GNN에서 맥락·특징·관계 3가지 불일치 문제를 정의하고 분석함.[1]
- 각 불일치 문제를 해결하기 위한 세 가지 모듈(컨텍스트 임베딩, 이웃 필터링 및 샘플링, 관계 주의(attention))을 통합한 **GraphConsis** 설계.[1]
- YelpChi 데이터셋 기반 실험에서 기존 GNN 및 비-GNN 기법 대비 F1 및 AUC 성능을 유의미하게 향상함.[1]

# 문제 정의

사기 탐지 그래프에서 GNN이 사용하는 이웃 집계(aggregation)는 이웃 노드들이 유사한 맥락·특징·관계를 가질 것이라는 가정에 기반한다. 하지만 실제 사기 그래프에서는 다음과 같은 **세 가지 불일치**가 발생한다:[1]

1. **맥락 불일치(Context inconsistency):**  
   - 사기 노드가 정상 노드와 다수 연결되어 위장(camouflage)함.  
   - 이로 인해 정상 이웃 정보를 집계하면 사기 노드를 구분하기 어려워짐.[1]

2. **특징 불일치(Feature inconsistency):**  
   - 동일 관계라도 노드 특성이 크게 다른 경우(ex. 같은 사용자 리뷰라도 제품 카테고리가 달라 텍스트 임베딩 차이 큼).  
   - 모든 이웃을 동등하게 집계하면 모델이 사기 특성을 학습하기 어려움.[1]

3. **관계 불일치(Relation inconsistency):**  
   - 여러 관계 유형(예: 사용자-리뷰, 제품-리뷰, 시간-리뷰)마다 이웃의 중요도가 다름.  
   - 관계를 일률적으로 처리하면 정보 손실 및 잡음이 유입됨.[1]

# 제안 기법: GraphConsis

GraphConsis는 이 세 가지 불일치를 완화하기 위해 다음 세 모듈을 차례로 적용한다:[1]

1. **컨텍스트 임베딩(Context Embedding):**  
   - 각 노드 $$v$$에 학습 가능한 컨텍스트 벡터 $$\mathbf{c}_v$$를 추가 도입.  
   - 1층 집계 시 노드 특성 $$\mathbf{x}_v$$를 $$\mathbf{c}_v$$와 연결(concatenation)하여 입력:  

```math
       \mathbf{h}_v^{(1)} = [\,\mathbf{x}_v\Vert \mathbf{c}_v ] \oplus \text{AGG}^{(1)}\bigl\{[\mathbf{x}_{v'}\Vert \mathbf{c}_{v'}]:v'\in \mathcal{N}(v)\bigr\}.
```

2. **이웃 샘플링(Neighbor Sampling):**  
   - 층 $$l$$에서 노드 $$u,v$$ 간 임베딩 거리 기반 일치도 점수:

$$
       s^{(l)}(u,v) = \exp\bigl(-\|\mathbf{h}_u^{(l)} - \mathbf{h}_v^{(l)}\|_2^2\bigr).
     $$
  
   - 임계값 $$\epsilon$$ 이하의 이웃은 제거 후, 나머지 이웃을 점수 정규화로 샘플링 확률 할당:

$$
       p^{(l)}(u;v) = \frac{s^{(l)}(u,v)}{\sum_{u'\in \tilde{\mathcal{N}}(v)}s^{(l)}(u',v)}.
     $$

3. **관계 주의(Relation Attention):**  
   - 각 관계 $$r$$마다 학습 가능한 벡터 $$\mathbf{t}_r$$ 도입.  
   - 샘플링된 이웃 $$q=1,\dots,Q$$에 self-attention 적용:

$$
       \alpha_q^{(l)} = \frac{\exp\bigl(\sigma\bigl([\,\mathbf{h}_q^{(l)}\Vert\mathbf{t}_{r_q}]\mathbf{a}^\top\bigr)\bigr)}
         {\sum_{q'=1}^Q\exp\bigl(\sigma\bigl([\,\mathbf{h}_{q'}^{(l)}\Vert\mathbf{t}_{r_{q'}}]\mathbf{a}^\top\bigr)\bigr)},
     $$
   
   - 가중합으로 이웃 임베딩 집계:

$$
       \text{AGG}^{(l)} = \sum_{q=1}^Q\alpha_q^{(l)}\,\mathbf{h}_q^{(l)}.
     $$

이 과정을 통해 **맥락 불일치**는 노드별 컨텍스트 임베딩으로, **특징 불일치**는 일치도 기반 샘플링으로, **관계 불일치**는 관계 주의로 각각 완화한다.[1]

# 모델 구조

GraphConsis는 이종(heterogeneous) 그래프 GNN으로,  
- 입력: 노드 특성, 3종 관계(사용자·제품·시간)  
- 은닉층: 2-layer GNN  
  - 1층 샘플 수 10, 2층 샘플 수 5  
  - 은닉 차원 200→100  
- 출력: 노드 분류(사기 vs 정상)  
- 손실: 크로스엔트로피, 옵티마이저 Adam  
- 평가: F1-score, AUC[1]

# 성능 향상 및 한계

**실험 결과 (YelpChi 스팸 리뷰 분류)**  
- GraphConsis는 40%~80% 학습 데이터 비율에서 F1 및 AUC 모두 기존 기법을 앞섬.  
  - 예: 80% 학습 시 F1=0.5776, AUC=0.7428로 최상.[1]
- 비-GNN Logistic Regression의 안정적 AUC 대비, GNN만 사용할 경우 일관되게 낮은 AUC 관찰 → 이웃 집계의 불일치 문제 심각성 확인.[1]
- Player2Vec(관계 주의만 적용) 대비 성능 우위 → 관계만 고려해서는 불충분함을 시사.[1]

**한계 및 고려 사항:**  
- 현재 임계값 $$\epsilon$$를 고정 사용 → 데이터별 적응형 임계값 학습 필요  
- 실험 데이터셋 한정(YelpChi) → 금융 사기, 사이버 범죄 등 다른 도메인 검증 필요  
- 노드 수·차원 증가 시 계산 비용 상승 → 효율적 샘플링·차원 축소 연구 필요.[1]

# 모델 일반화 성능 향상 가능성

GraphConsis의 **일치도 기반 샘플링**은 이웃 간 불필요한 잡음 노드를 제거하여, **다양한 그래프 도메인**에서 과적합을 완화하고 일반화 성능을 높일 가능성이 있다. 특히,  
- **소수 클래스(사기) 희소성** 문제 완화: 불일치 노드 배제로 사기 노드 표현 학습 집중  
- **다중 관계**를 갖는 복잡 네트워크(금융·통신·소셜)에 적용 시, 관계별 중요도 반영으로 과적합 방지  

따라서, GraphConsis 구조는 **범용 GNN 불일치 완화** 모듈로 확장 가능하다.

# 향후 연구 영향 및 고려 사항

**연구 영향:**  
- GNN 기반 이상 탐지(fraud/anomaly detection) 분야에서 **이불일치(inconsistency)** 개념 조명  
- 후속 연구에서 **적응형 샘플링 임계값**, **다중 도메인 검증**, **효율적 관계 주의** 기법 개발 촉진  

**앞으로 고려할 점:**  
- **임계값 자동 학습**: 각 관계·층별 최적 $$\epsilon$$ 탐색  
- **대규모 그래프 확장성**: 분산 샘플링·경량화 모델 설계  
- **다양한 도메인 실험**: 금융 거래, 네트워크 보안 로그 등 복합 데이터셋 적용  

위와 같은 방향을 통해 GraphConsis의 일반화 성능과 실용 가치를 더욱 높일 수 있을 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d7b92742-a452-44e1-a6fe-9b1e002c05aa/2005.00625v3.pdf)
