# Benchmarking Graph Neural Networks

**핵심 주장 및 주요 기여**  
본 논문은 그래프 신경망(Graph Neural Networks, GNN) 분야의 **공정하고 재현 가능한 벤치마크 프레임워크**를 제안한다. 이 프레임워크는  
- 다양한 수학적·실세계 그래프 데이터셋 수집  
- 동일 파라미터 예산으로 모델 간 공정 비교  
- 오픈소스화된 모듈화 코드 인프라 제공  
- 연구자가 새로운 이론 아이디어를 실험할 수 있는 유연성  
등을 통해 GNN 연구의 **진행 상황을 정량화**하고 **핵심 아키텍처와 설계 원칙**을 식별할 수 있도록 지원한다.[1]

***

## 해결하고자 하는 문제  
기존 GNN 연구는  
1. 데이터셋 규모 및 도메인이 제한적  
2. 실험 설정과 하이퍼파라미터가 비일관적  
3. 모델 간 파라미터 규모 차이로 성능 비교의 공정성 결여  
등으로 인해 새로운 아키텍처나 메커니즘의 진정한 가치를 판별하기 어려웠다.[1]

***

## 제안 방법  
### 벤치마크 프레임워크 구성  
1. **데이터 모듈**: 12개 대표 데이터셋(ZINC, AQSOL, OGBL-COLLAB, WikiCS, MNIST, CIFAR10, PATTERN, CLUSTER, TSP, CSL, CYCLES, GraphTheoryProp)  
2. **구성(config) 모듈**: 모델·데이터·하이퍼파라미터 설정 파일  
3. **레이어(layers) 모듈**: MP-GCN, GIN, GAT, GraphSage, MoNet, GatedGCN, RingGNN, 3WL-GNN 등 다양한 GNN 레이어 정의  
4. **네트워크(nets) 모듈**: 레이어 조합으로 완전한 GNN 아키텍처 구성  
5. **학습(train) 모듈**: 공통 학습·평가 스크립트  

이 모든 모듈은 PyTorch/DGL 기반으로 개발되어, 동일 파라미터 예산(100K, 500K) 하 공정 비교를 보장한다.[1]

### 수식 요약  
- 일반 MP-GCN 레이어:  

$$
h_i^{(\ell+1)} = \sigma\Bigl(U^{(\ell)}\frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)}h_j^{(\ell)}\Bigr)
$$

- GAT 다중 헤드 어텐션:  

$$
h_i^{(\ell+1)} = \mathop{\Vert}_{k=1}^K \text{ELU}\Bigl(\sum_{j\in\mathcal{N}(i)}\alpha_{ij}^{(k)}W^{(k)}h_j^{(\ell)}\Bigr)
$$

$$
\alpha_{ij}^{(k)} = \frac{\exp\bigl(\text{LeakyReLU}(a^{(k)\top}[Wh_i\Vert Wh_j])\bigr)}{\sum_{j'\in\mathcal{N}(i)}\exp(\dots)}
$$

- WL-GNN(3WL-GNN) 레이어(순열 불변):  

$$
H^{(\ell+1)} = \bigl(M_1(H^{(\ell)})\cdot M_2(H^{(\ell)}),\,M_3(H^{(\ell)})\bigr)
$$
  
***

## 모델 구조  
- **메시지 패싱 GCN 계열**: Kipf&Welling GCN, GraphSage, GAT, MoNet, GatedGCN, GIN  
- **윈저라이더-레만 계열**: RingGNN(2-WL), 3WL-GNN(3-WL)  
- **읽기·예측 레이어**: 그래프, 노드, 엣지 분류·회귀 용 MLP  

모든 모델은 배치 정규화, 잔차 연결, 동일 예산 파라미터(100K/500K) 제약을 통일적으로 적용한다.[1]

***

## 성능 향상 및 한계  
- **MP-GCN 우위**: 다양한 중규모 데이터셋(ZINC, AQSOL, TSP 등)에서 메시지 패싱 기반 모델이 3WL-GNN보다 평균적으로 우수한 성능을 보였다.[1]
- **계산 복잡도**: WL-GNN(3WL, RingGNN)은 dense 텐서·순열 불변성을 유지하나, 메모리 사용량(O(n³)) 과 배치 불가능성으로 인해 중·대규모 데이터셋에서 OOM 발생.[1]
- **공정 비교**: 동일 파라미터 예산 하에서도 모델 간 성능 차이가 아키텍처 설계의 본질적 차이임을 입증했다.  
- **한계**: 대규모 그래프, 서브그래프 배치(normalization·batch training)는 여전히 GCN 계열에 의존.  

***

## 일반화 성능 향상 가능성  
논문에서는 **위치 인코딩(Positional Encoding, PE)** 개념을 도입해 MP-GCN의 이론적 한계를 보완하였다.  
- **Laplacian PE**: 그래프 라플라시안 고유 벡터 $$U$$의 처음 k개 비자명(eigenvectors, non-trivial)으로 노드별 위치 표현 추가  
- **수식**:  
$$
\Delta = I - D^{-1/2}AD^{-1/2} = U\Lambda U^\top,\quad p_i = [U_{i,2},\dots,U_{i,k+1}]
$$
이후 원래 특징 $$x_i$$에 $$p_i$$를 더해 입력함으로써, **동형 그래프 구분 능력** 및 **일반화 성능**이 크게 향상됨을 보였다.  
- **결과**: CSL, CYCLES 등 이론적 검증용 데이터에서 PE 적용 시 1-WL 한계를 극복하고 100% 정확도 달성.[1]

***

## 앞으로의 영향 및 연구 고려사항  
- **기준 벤치마크 정착**: GNN 연구 전반에 걸쳐 공통 실험 프로토콜과 데이터셋을 제공해, 새 아키텍처 검증의 투명성과 신뢰성 향상  
- **위치 인코딩 확장**: Laplacian PE 외에도 스펙트럴·거리 기반 PE, Learnable PE 연구 활발화  
- **대규모·산업용 그래프**: 현재 제한적인 중규모 실험을 넘어, 매우 큰 그래프(수백만 노드)에 대한 **배치 학습**·**효율적 정규화**·**저메모리 모델** 개발 필요  
- **이론적·실용적 균형**: ‘표준 예산’ 하 이론적 표현력 확보와 실전 응용 간 트레이드오프 최적화 연구  

Benchmarked GNN 프레임워크는 향후 GNN 설계의 **첫 원칙(first principles)** 도출과 **확장 가능·공정한** 비교 연구에 중대한 기반이 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/23aba2f3-f5be-4dc6-aca4-6995fd7b1921/2003.00982v5.pdf)
