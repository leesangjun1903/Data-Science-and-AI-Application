# node2vec: Scalable Feature Learning for Networks

## 1. 간결 요약
**핵심 주장**  
node2vec는 그래프 데이터를 위한 유연한 이웃 탐색 전략을 통해 노드 임베딩을 학습하는, 확장성 높은 반지도 학습 프레임워크이다. 기존의 BFS/DFS 기반 임베딩 기법을 일반화하여 두 방식 사이를 매개하는 파라미터 $$p, q$$를 도입함으로써 네트워크의 **동질성(homophily)**과 **구조적 역할(structural equivalence)**을 모두 효과적으로 포착한다.[1]

**주요 기여**  
- 2차 마르코프 편향 확률 기반 랜덤 워크로 BFS·DFS 탐색을 연속적으로 조절  
- Negative sampling을 활용한 효율적 SGD 최적화로 대규모 네트워크에도 적용 가능  
- 노드 임베딩을 간단한 이진 연산자(평균, Hadamard, L1/L2 거리)로 확장하여 엣지 표현 학습 지원[1]

## 2. 문제 정의 및 제안 방법

### 2.1 해결하고자 하는 문제  
- 네트워크 분석(노드 분류·링크 예측 등)을 위해 수작업 특징 공학이 필요하며 범용적으로 재사용 어려움  
- 기존 차원 축소 기반 기법(PCA, IsoMap)과 단일 탐색 전략(DeepWalk, LINE)은  
  - 네트워크의 **다양한 연결 패턴**(community vs. role)에 대한 표현력이 부족  
  - 대규모 네트워크에 대한 **확장성** 및 **표본 재사용** 미흡  

### 2.2 제안 방법: node2vec

#### 목적 함수  

$$
\max_{f} \sum_{u \in V} \log \Pr\bigl(N_S(u)\mid f(u)\bigr)
$$  

- $$N_S(u)$$: 노드 $$u$$의 샘플링된 이웃 집합  
- Conditional independence 및 softmax 가정 하에  

$$
\Pr(n_i\mid f(u)) = \frac{\exp\bigl(f(n_i)\cdot f(u)\bigr)}{\sum_{v\in V}\exp\bigl(f(v)\cdot f(u)\bigr)}
$$  

- Negative sampling으로 partition function 근사[1]

#### 탐색 전략: 2차 마르코프 랜덤 워크  
- **Return parameter** $$p$$: 직전 노드로의 재방문 확률 제어  
- **In–out parameter** $$q$$: 근접/원격 노드 탐색 편향  
- transition probability  

$$
\pi_{vx} = \alpha_{pq}(t,x)\cdot w_{vx},\quad 
\alpha_{pq}(t,x)=
\begin{cases}
\frac1p & d_{tx}=0,\\
1 & d_{tx}=1,\\
\frac1q & d_{tx}=2,
\end{cases}
$$  

- 이로써 BFS⇄DFS 사이 탐색 연속 조절 가능[1]

#### 모델 구조  
1. **Preprocessing**: 모든 엣지 $$(v,x)$$에 대해 $$\pi_{vx}$$ 계산  
2. **Random Walk 시뮬레이션**: 각 노드마다 길이 $$l$$, 횟수 $$r$$만큼 워크 수행  
3. **SGD 최적화**: Skip-gram 유사 구조로 negative sampling 활용  

#### 엣지 표현 학습  
- 노드 임베딩 $$f(u), f(v)$$에 대해 이진 연산자 $$\circ$$ 적용  
  - 평균, Hadamard, Weighted-L1, Weighted-L2 연산자 중 선택[1]

### 2.3 성능 향상 및 한계

#### 성능  
- **노드 분류**: BlogCatalog에서 DeepWalk 대비 Macro-F1 22.3%↑, LINE 대비 229.2%↑[1]
- **링크 예측**: arXiv 네트워크에서 Adamic-Adar 대비 AUC 12.6%↑[1]
- **확장성**: 1백만 노드까지 선형 스케일, 4시간 이내 처리[1]
- **강건성**: 노드 이웃 손실 및 잡음 엣지에도 완만한 성능 감소[1]

#### 한계  
- 하이퍼파라미터 $$p, q$$ 탐색 비용  
- 워크 길이·횟수 증가 시 샘플링·최적화 부하 상승  
- 이진 연산자의 선택에 따라 링크 예측 편차 발생  

## 3. 일반화 성능 향상 관점
node2vec의 **튜닝 가능한 탐색 파라미터**는 네트워크마다 최적의 커뮤니티 구조와 구조적 역할을 동시에 반영할 수 있도록 한다.  
- **Semi-supervised** 방식으로 $$p, q$$를 적은 레이블(10%) 데이터를 통해 교차 검증으로 학습하여 **일반화 성능** 극대화  
- BFS·DFS 고정 전략 대비 다양한 연결 패턴을 학습 시 노드간 관계 특성이 잘 보존되어 미지의 네트워크에서도 강건  

## 4. 향후 연구에 미치는 영향 및 고려 사항
- **향후 영향**: node2vec는 이후 **GraphSAGE**, **GNN** 등 후속 그래프 신경망 연구의 기초 임베딩 방법으로 자리매김  
- **연구 시 고려할 점**:  
  1. **이종 정보 네트워크(heterogeneous networks)**, **속성 포함 네트워크**에 대한 탐색 편향 확장  
  2. **Signed-edge**나 **동적 그래프** 등 특수 네트워크 구조로 일반화  
  3. 엣지 임베딩 해석력 강화 및 **연산자 설계** 최적화  
  4. **하이퍼파라미터 자동 최적화** 기법 도입으로 사용자 부담 저감  

 node2vec: Scalable Feature Learning for Networks (attached_file:1)[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/464cbd61-3dec-4148-803b-32761a54d1e6/1607.00653v1.pdf
