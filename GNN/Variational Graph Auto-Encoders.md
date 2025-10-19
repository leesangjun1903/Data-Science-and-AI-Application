# Variational Graph Auto-Encoders

# 핵심 요약  
**Variational Graph Auto-Encoders(VGAE)**는 그래프 구조화된 데이터를 위한 **비지도 학습(latent variable model)** 프레임워크로, 그래프 컨볼루션 네트워크(Graph Convolutional Network, GCN)를 인코더로, 잠재 표현 간 단순 내적(inner product) 기반 디코더를 사용하는 것이 핵심 기여이다. 입력 노드 특징(node features)을 자연스럽게 통합하여 기존 비지도 그래프 임베딩 모델 대비 **링크 예측(link prediction)** 성능을 크게 향상시켰다.[1]

# 1. 해결하고자 하는 문제  
전통적인 그래프 비지도 학습 방법들은  
- **노드 간 구조 정보**만을 활용하거나  
- **노드 특징(X)**을 효과적으로 융합하지 못하는 한계  
를 지녔다.  
VGAE는 이 둘을 모두 고려하여, 잠재변수 기반 생성 모델로 그래프 재구성 능력을 학습함으로써, **희소한 그래프에서도 일반화된 잠재 표현**을 학습하고자 한다.  

# 2. 제안 방법  
## 2.1 인퍼런스 모델 (GCN 기반 인코더)  
잠재변수 $$Z\in\mathbb{R}^{N\times F}$$에 대한 근사 후방분포를 다음과 같이 정의한다:  

$$
q(Z\mid X,A)=\prod_{i=1}^N q(z_i\mid X,A),\quad
q(z_i\mid X,A)=\mathcal{N}(z_i\mid\boldsymbol\mu_i,\text{diag}(\boldsymbol\sigma_i^2))
$$  

여기서  

$$
\boldsymbol\mu = \mathrm{GCN}_\mu(X,A),\quad
\log \boldsymbol\sigma = \mathrm{GCN}_\sigma(X,A)
$$  

이고,  

$$
\mathrm{GCN}(X,A)=\widetilde{A}\,\mathrm{ReLU}(\widetilde{A}XW^{(0)})W^{(1)},\quad
\widetilde{A}=D^{-\frac12}AD^{-\frac12}
$$  

는 대칭 정규화 인접행렬을 이용한 두 층 GCN 구조이다.[1]

## 2.2 생성 모델 (Inner Product 디코더)  
잠재 표현 $$Z$$로부터 인접행렬 $$A$$를 재구성하기 위해 노드 쌍 $$(i,j)$$ 간의 연결 확률을  

$$
p(A_{ij}=1\mid z_i,z_j)=\sigma\bigl(z_i^\top z_j\bigr)
$$  

로 모델링한다. 여기서 $$\sigma(\cdot)$$는 로지스틱 시그모이드 함수이다.[1]

## 2.3 학습  
변분 하한(ELBO)을 최대화하여 인코더와 디코더 매개변수를 학습한다:  

$$
\mathcal{L} = \mathbb{E}_{q(Z\mid X,A)}\bigl[\log p(A\mid Z)\bigr]
- \mathrm{KL}\bigl[q(Z\mid X,A)\,\|\,p(Z)\bigr],
$$  

우선 가우시안 사전분포 $$p(Z)=\prod_i\mathcal{N}(0,I)$$를 사용하며, 재파라미터화 트릭을 활용해 경사하강법으로 최적화한다.[1]

# 3. 모델 구조  
- **인코더**: 2-layer GCN with ReLU  
  - 1st layer: $$X\to H=\mathrm{ReLU}(\widetilde{A}XW^{(0)})$$  
  - 2nd layer: $$H\to \{\boldsymbol\mu,\log\boldsymbol\sigma\}=H W^{(1)}$$  
- **디코더**: inner product 기반 확률 출력  
- **변분 학습**: ELBO 최적화, 재파라미터화 트릭  

# 4. 성능 향상 및 한계  
- **입력 특징 통합** 시 기존 Spectral Clustering, DeepWalk 대비 링크 예측 AUC, AP가 대폭 개선되었으며, 특히 Cora·Citeseer·Pubmed 데이터셋에서 4~10% 성능 상승을 보였다.[1]
- **한계**  
  - 가우시안 사전분포와 단순 내적 디코더 조합이 잠재공간을 원점 중심으로 강제하지 못해 표현력이 제약될 수 있음.  
  - 대규모 그래프 적용 시 전체 배치 학습(full-batch)으로 인한 확장성 문제 존재.  

# 5. 일반화 성능 향상 관점  
VGAE의 **잠재 변수 기반 생성 모델**은 노드 간 구조와 특징을 통합 학습하여,  
- **희소 연결**에서도 노드 관계를 포착  
- **잠재 표현**의 **부드러운 공간 구조**를 학습  
함으로써, **미관측 링크 예측** 및 **노드 분류 등 하류 작업에 대한 일반화 성능**을 향상시킨다.  

# 6. 향후 연구 영향 및 고려 사항  
- **더 유연한 사전분포(prior)**: 비가우시안 분포(예: VMF, mixture 모델) 적용으로 잠재공간 구조 개선  
- **확장성**: 그래프 샘플링(GraphSAGE) 또는 무작위 배치 변분 추론으로 대규모 그래프 지원  
- **디코더 구조**: 내적 외에도 신경망 기반 디코더로 비선형 재구성 모델 개발  
- **정규화 기법**: 잠재공간에 대한 추가 제약(예: 분산 제어)으로 과적합 방지 및 일반화 강화  

이 논문은 **그래프 딥러닝** 분야에서 변분 생성 모델의 가능성을 제시하며, 향후 **그래프 잠재 표현 학습**과 **딥 그래프 생성 모델** 연구에 중요한 기반을 제공할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/dbb87e4a-813c-472b-91b2-c2a6c2d4cb77/1611.07308v1.pdf)
