# Self-Organizing Maps 가이드

핵심 요약: Self-Organizing Map(SOM)은 고차원 데이터를 저차원 격자에 투영하면서 데이터의 **위상**을 보존하도록 학습하는 경쟁학습 기반의 신경망입니다. 모델은 BMU(최적일치노드)와 이웃(neighborhood) 업데이트로 학습되며, 시각화·군집화·표현학습에 유용합니다.[1][2]

### best matching unit (BMU)
BMU(최적일치노드)는 자기조직화지도(SOM)에서 입력 벡터와 가장 유사한 가중치 벡터를 가진 노드를 의미합니다.  
보통 입력 벡터와 각 노드 가중치 벡터 간의 유클리드 거리를 계산하여, 가장 거리가 가까운 노드를 BMU로 선정합니다.

즉, BMU는 입력 데이터와 가장 잘 맞는 노드로서, SOM 학습 과정에서 해당 노드와 이웃 노드의 가중치가 입력 데이터에 맞게 조정되어 군집화와 차원 축소가 이루어집니다.

## 무엇이 SOM인가
SOM은 입력 공간의 유사도 구조를 2D 격자(map) 위에 보존하도록 정렬하는 비지도 신경망입니다. 가까운 데이터는 격자에서도 가깝게 배치되며, 멀리 있는 데이터는 떨어지도록 배웁니다. 이 과정은 경쟁학습과 이웃-보정 규칙으로 이루어지며, 오류역전파를 쓰지 않습니다.[1]

## 핵심 직관
각 격자 노드에는 입력공간의 좌표를 뜻하는 가중치 벡터가 있습니다. 입력이 들어오면 모든 노드와의 거리(보통 유클리드)를 계산하고, 가장 가까운 BMU와 그 주변 이웃의 가중치를 입력 쪽으로 조금씩 이동시킵니다. 이때 이동 크기와 이웃 범위는 시간에 따라 점감합니다.[2]

## 수식으로 이해하기
입력 벡터를 $$x$$, BMU를 $$c$$, 노드 $$j$$의 가중치를 $$w_j$$라 하면 한 스텝 업데이트는 다음과 같습니다.[2]

$$ w_j \leftarrow w_j + \alpha(t)\theta(\|r_j-r_c\|,t)(x - w_j) $$  

여기서 $$r_j$$는 격자 좌표, $$\alpha(t)$$는 학습률(단조 감소), $$\theta$$는 이웃 함수(거리와 시간에 따라 감소)입니다. 흔한 선택은 가우시안 이웃 $$\theta=\exp(-\|r_j-r_c\|^2/2\sigma(t)^2)$$이며  $$\sigma(t)$$도 점감시킵니다 [2][3][4].

## SOM vs. 벡터양자화/군집
SOM은 k-means와 유사하게 코드북(센트로이드)을 학습하지만, 격자 위상을 강제해 코드북 간 전역적 순서를 형성합니다. 이 덕분에 지도는 연속적인 데이터 매니폴드의 구조를 더 잘 드러내고, U-Matrix 등으로 경계·밀도를 시각화할 수 있습니다. 위상 제약이 없는 순수 VQ보다 시각적 해석성이 높습니다.[1]

**SOM(Self-Organizing Map)**은 고차원 데이터를 2차원 평면 격자에 투영하여 시각화하고 군집화하는 인공신경망 기반의 차원 축소 및 군집화 기법입니다.  
입력 벡터와 노드(뉴런)들의 가중치 벡터 간 거리를 계산하여 가장 유사한 노드(BMU)를 찾고, 그 노드와 주변 노드들의 가중치를 입력 벡터에 가까워지도록 업데이트합니다.

반면, **벡터 양자화(Vector Quantization)**는 고차원 벡터 데이터를 유사한 벡터들로 묶어 대표 벡터(코드북)를 만들어 저장 공간과 계산 자원을 절약하는 압축 기술입니다.  
벡터 양자화를 통해 메모리 사용량을 줄이고 계산 효율성을 높이며, 특히 대규모 검색과 생성형 AI에 활용됩니다.

## 설계 선택지
- 격자 구조: 직사각형/육각형, 1D/2D(연구·시각화는 2D가 일반적).[5]
- 이웃 함수: 가우시안이 표준, 멕시칸햇·삼각형 등 변형도 연구됨.[6][2]
- 스케줄: $$\alpha(t)$$, $$\sigma(t)$$는 크고-넓게 시작해 점감하며 수렴을 유도합니다.[3][2]

## 초기화 전략
무작위 초기화는 보편적입니다. PCA 주성분 공간에서 초기화하면 수렴이 빠르고 재현성이 좋습니다. 데이터 기하에 따라 초깃값의 이점이 달라질 수 있어, 실험적으로 선택하는 것이 안전합니다.[1]

## 학습 절차(알고리즘)
- 모든 노드 가중치를 초기화합니다. 무작위 또는 PCA 기반 초기화가 쓰입니다.[1]
- 각 스텝에서 입력 $$x$$를 샘플링하고, BMU를 찾습니다(최소 거리).[1]
- BMU와 이웃 노드의 가중치를 $$x$$ 방향으로 업데이트합니다. $$\alpha,\sigma$$는 점감합니다.[2]
- 충분한 에폭 동안 반복하여 위상 정렬을 형성합니다.[2]

## 왜 동작하는가(해석)
이웃 전파는 지역적으로 매끄러운 코드북을 만들며, 이는 고차원 데이터의 연속적 구조를 저차원 격자상에 펼칩니다. 결과적으로 유사한 샘플이 인접 뉴런을 흥분시키고, 격자는 데이터 분포의 비연속적 근사로 작동합니다. 이러한 위상 보존은 생물학적 맵핑(피질 지형도)에서 영감을 받았습니다.[1]

## GTM과의 비교
GTM은 SOM을 확률모형으로 재정식화한 생성모형입니다. 잠재 저차원 격자에서 비선형 사상(RBF 등)으로 데이터 공간으로 맵핑하고, 가우시안 잡음을 가정해 EM으로 학습합니다. 수렴성이 좋고, 수축 이웃·감소 스텝을 필수로 요구하지 않는 장점이 있습니다. 확률적 해석과 불확실성 다루기가 필요하면 GTM을 고려합니다.[7][8][9][10]

### Generative Topographic Map (GTM)
Generative Topographic Map (GTM)은 자가 조직화 지도(Self-Organizing Map, SOM)의 확률적 대응 모델로, 저차원 잠재공간에서 점을 선택해 이를 고차원 관측공간으로 부드럽게 매핑하는 생성적 기계 학습 방법입니다.  
매개변수는 기대 최대화(EM) 알고리즘으로 학습되며, 비선형 매핑에는 주로 RBF(방사 기저 함수) 네트워크가 이용됩니다.

GTM은 자기조직화지도(SOM)의 확률적 대응 방법으로, 데이터가 저차원 공간의 잠재 변수 $(\mathbf{x})$에서 확률적으로 선택되고, 비선형 함수 $(\mathbf{y}(\mathbf{x}; \mathbf{W}))$에 의해 고차원 관측 공간으로 매핑된 후, 가우시안 노이즈가 더해진다고 가정합니다.

수식으로 표현하면 다음과 같습니다:

```math
[p(\mathbf{t}|\mathbf{x}, \mathbf{W}, \beta) = \left(\frac{\beta}{2\pi}\right)^{D/2} \exp\left(-\frac{\beta}{2}|\mathbf{t} - \mathbf{y}(\mathbf{x}; \mathbf{W})|^2\right)
]
```

여기서

- $(\mathbf{t})$는 관측 데이터 벡터 (고차원 공간),
- $(\mathbf{x})$는 잠재 변수 (저차원 공간),
- $(\mathbf{y}(\mathbf{x}; \mathbf{W}))$는 매핑 함수이며, 보통 RBF(방사 기저 함수)와 가중치 행렬 $(\mathbf{W})$로 표현됩니다,
- $(\beta)$ 는 가우시안 노이즈의 역분산(정확도 파라미터),
- (D)는 데이터 차원입니다.

매핑 함수 $(\mathbf{y}(\mathbf{x}; \mathbf{W}))$는 보통 아래와 같은 형태를 가집니다:

```math
[\mathbf{y}(\mathbf{x}; \mathbf{W}) = \mathbf{W} \boldsymbol{\phi}(\mathbf{x})
]
```

여기서 $(\boldsymbol{\phi}(\mathbf{x}))$ 는 RBF 기저 함수들의 벡터이며, $(\mathbf{W})$ 는 각 RBF의 출력에 대한 가중치 행렬입니다.

학습은 EM 알고리즘으로 수행하며, 이산 격자 형태의 잠재 공간에서 각 잠재 점들이 데이터에 대한 가우시안 혼합 모델의 중심이 되고, 데이터 지점들은 이 혼합 모델에 의해 설명됩니다. 이 과정에서 모수 $(\mathbf{W}), (\beta)$, 그리고 잠재 공간 확률 분포가 갱신되어 데이터에 최적화됩니다.

#### Radial Basis Function, RBF
방사 기저 함수(Radial Basis Function, RBF)는 입력과 특정 기준점(랜드마크) 사이의 거리에 따라 값이 변하는 함수로, 주로 가우시안 함수 형태로 정의됩니다. 수식은 일반적으로 다음과 같습니다.

```math
[\varphi(x) = \exp(-\gamma | x - c |^2)
]
```

여기서

- (x)는 입력 벡터
- (c)는 랜드마크 또는 중심점
- $(\gamma)$는 폭 조절 파라미터(양수)
- $(| x - c |)$는 (x)와 (c) 사이의 유클리드 거리입니다.

이 함수는 (x)가 랜드마크 (c)에 가까울수록 1에 가까워지고, 멀어질수록 0에 수렴하는 종 모양(가우시안 형태)의 값 변화를 보입니다.  
RBF는 비선형 데이터 분류, 함수 근사, 신경망 활성화 등에 활용됩니다. 특히, SVM의 커널 함수로도 널리 쓰이며 복잡한 비선형 패턴을 모델링하는 데 효과적입니다.

주요 특징은 다음과 같습니다:

GTM은 데이터 분포를 가우시안 혼합모형으로 모델링하며, 데이터 생성 과정을 확률적으로 설명합니다.  
SOM과 달리 수렴성이 보장되고, 이웃 크기 축소나 감소하는 학습률이 필요 없습니다.  
비선형 차원 축소뿐 아니라 분류, 회귀 등에도 확장이 가능합니다.  
확률적 모델이라서 명확한 비용 함수와 과적합 방지, 맵의 해석이 가능하며, 트리구조 같은 복잡한 데이터에도 응용됩니다.  
변형된 형태로 구면 위의 매핑 등을 통한 화학 데이터 시각화 등 다양한 응용도 연구되고 있습니다.  

요약하면, GTM은 데이터의 저차원 잠재구조를 확률적 관점에서 모델링해 데이터 시각화, 차원 축소, 분류 등에 활용하는 강력한 기계학습 기법입니다.

## 실제 적용 포인트
- 데이터 스케일: 표준화가 BMU 탐색 안정성과 수렴에 유리합니다.[1]
- 맵 크기: 샘플 수 대비 충분한 노드 수가 양자화 오차를 줄입니다. 과도하면 과적합·희소 활성 문제가 생길 수 있습니다.[1]
- 품질 지표: 양자화 오차(QE), 위상 보존/토포그래픽 에러(TE), U-Matrix 시각화로 점검합니다.[1]
- 변형: 가변 이웃, 적응형 학습률, 승자빈도 보정 등 대규모·고차원 개선안이 제안되어 왔습니다.[11][12][6]

### QE, Quantization Error (유클리드 거리 기반)
```math
[QE = \frac{1}{N} \sum_{i=1}^N | X_i - m_{c_i} |
]
```

- (N): 입력 벡터 수 (예: 이미지 픽셀 개수)
- $(X_i)$: i번째 입력 벡터
- $(m_{c_i})$ : i번째 입력에 가장 가까운 모델 벡터(프로토타입)
이는 입력과 근사된 출력 간 거리의 평균을 뜻합니다.

### Topographic Error, TE
TE는 인접한 노드들이 실제 입력 데이터에서도 인접하는지를 측정하며, 보통 다음 수식으로 정의됩니다.

토포그래픽 에러 지표 수식 :
```math
[TE = \frac{1}{N} \sum_{i=1}^{N} u(x_i)
]
```

여기서

- (N)은 입력 데이터 샘플의 총 개수
- $(u(x_i))$는 입력 벡터 $(x_i)$에 대해 1번째로 가까운 BMU(best matching unit)와 2번째로 가까운 BMU가 서로 인접해 있으면 0, 그렇지 않으면 1을 의미합니다. 즉, 이 두 BMU가 인접하지 않으면 토포그래픽 에러가 발생한 것으로 판단합니다.

이 지표는 SOM이 데이터의 위상적 배열을 얼마나 잘 유지하는지, 즉 입력 공간의 근접성이 출력 맵에서도 유지되는지를 평가하는 데 쓰입니다. 값이 0에 가까울수록 위상 보존 성능이 뛰어납니다.

### U-Matrix 시각화
U-Matrix 시각화는 Self-Organizing Map(SOM)에서 클러스터 경계를 시각화하기 위한 기법으로, 인접한 SOM 유닛 간의 거리(불일치도)를 계산해 이를 지도 격자 위에 색상 또는 높이 차원으로 표현하는 방법입니다.

주요 내용은 다음과 같습니다:

- 원리: 각 SOM 유닛과 그 주변 이웃 유닛들 간의 프로토타입 벡터 간 거리를 계산하여 행렬로 나타냅니다. 이 거리 값들을 색상 또는 3D 높이 형태로 표현해, 값이 클수록 클러스터 경계임을 시사합니다.

프로토타입 벡터는 SOM 각 노드가 대표하는 고차원 입력 데이터의 중심점 벡터이며, U-Matrix는 이 프로토타입 벡터 간 거리를 측정해 육각형 또는 격자 형태로 표현하는 2차원 시각화 도구입니다.  
이를 통해 데이터 군집 구조 및 경계를 파악할 수 있습니다.

U-Matrix 크기는 보통 SOM 출력이 n×m일 때 (2n-1)×(2m-1) 크기의 그래프로 확대되어, 노드와 노드 사이 거리를 중간 공간에 표현합니다.  
보통 R의 Kohonen 패키지나 Python MiniSom같은 라이브러리를 통해 자동으로 생성 및 시각화합니다.

- 목적: 데이터 내 군집의 구조적 경계를 쉽게 식별하고, 데이터 토폴로지(위치 관계)를 파악하는 데 사용됩니다. 군집 내는 거리가 작아 어두운 색 등이, 경계는 거리가 커서 밝은 색 또는 높은 산 모양으로 표현됩니다.

- 확장: U-Matrix의 확장판인 U*-Matrix는 데이터 밀도를 함께 고려해 더 부드러운 군집 시각화를 제공하며, 대규모 SOM에서도 해석 가능성을 높입니다.

응용: 여러 고차원 데이터의 군집 분석, 이상치 탐지, 새로운 관측치 매핑 분석 등에 활용되며, R 패키지 등 도구로 구현되어 있습니다.

## 연구 수준 PyTorch 구현 예시
아래 코드는 연구 실험용 최소구현입니다. 배치 BMU 탐색, 가우시안 이웃, 선형 감소 스케줄을 포함합니다. 시각화는 U-Matrix와 코드북 맵핑을 전제로 합니다.[3][2]

```python
import torch
import math

class SOM(torch.nn.Module):
    def __init__(self, m, n, dim, sigma_start=None, sigma_end=1.0, lr_start=0.5, lr_end=0.05, device='cpu'):
        super().__init__()
        self.m, self.n, self.dim = m, n, dim
        self.num_nodes = m * n
        self.device = device
        self.lr_start, self.lr_end = lr_start, lr_end
        # 격자 좌표
        xs, ys = torch.meshgrid(torch.arange(m), torch.arange(n), indexing='ij')
        self.register_buffer('grid', torch.stack([xs, ys], dim=-1).view(-1, 2).float())  # (MN, 2)
        # 가중치 초기화: N(0,1) -> 정규화
        w = torch.randn(self.num_nodes, dim)
        w = w / (w.norm(dim=1, keepdim=True) + 1e-8)
        self.w = torch.nn.Parameter(w)
        # 초기 sigma
        if sigma_start is None:
            sigma_start = max(m, n) / 2.0
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end

    def forward(self, x):
        # x: (B, dim), BMU 인덱스 반환
        d2 = torch.cdist(x, self.w, p=2)  # (B, MN)
        bmu = d2.argmin(dim=1)  # (B,)
        return bmu

    def neighborhood(self, bmu_idx, sigma):
        # bmu_idx: (B,), grid: (MN,2)
        # 각 배치 BMU에 대한 가우시안 이웃 가중치 반환 (B, MN)
        bmu_coords = self.grid[bmu_idx]  # (B,2)
        # 거리 행렬: (B, MN)
        d2 = torch.cdist(bmu_coords, self.grid, p=2) ** 2
        h = torch.exp(-d2 / (2 * (sigma ** 2) + 1e-8))
        return h  # (B, MN)

    def step(self, x, t, T):
        # 스케줄
        lr = self.lr_start + (self.lr_end - self.lr_start) * (t / max(1, T-1))
        sigma = self.sigma_start + (self.sigma_end - self.sigma_start) * (t / max(1, T-1))
        # BMU
        with torch.no_grad():
            d2 = torch.cdist(x, self.w, p=2)              # (B, MN)
            bmu = d2.argmin(dim=1)                        # (B,)
            h = self.neighborhood(bmu, sigma)             # (B, MN)
            # 정규화된 이웃 가중치
            h = h / (h.sum(dim=1, keepdim=True) + 1e-8)   # (B, MN)
            # 업데이트: (B, MN, 1)*(B,1,dim)
            delta = (x.unsqueeze(1) - self.w.unsqueeze(0))  # (B, MN, dim)
            upd = (h.unsqueeze(-1) * delta).sum(dim=0)       # (MN, dim), 배치 합
            self.w.add_(lr * upd)

def train_som(data, m=20, n=20, epochs=20, batch_size=256, device='cpu'):
    data = data.to(device)
    som = SOM(m, n, data.shape[1], device=device).to(device)
    N = data.shape
    T = epochs
    for epoch in range(epochs):
        idx = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            x = data[idx[i:i+batch_size]]
            som.step(x, t=epoch, T=T)
    return som

# 사용 예시
# X: (N, D) 표준화 권장
# som = train_som(torch.from_numpy(X).float(), m=30, n=30, epochs=50, device='cuda')
# bmu = som(torch.from_numpy(X).float().to('cuda'))  # (N,)
```
이 구현은 배치별로 BMU를 찾고, 가우시안 이웃으로 업데이트를 누적합니다. $$\alpha(t)$$와 $$\sigma(t)$$는 선형 감소이며, 격자 거리에 기반한 $$\theta$$를 사용합니다. 실험 시에는 표준화, 충분한 에폭, U-Matrix·QE/TE 평가를 권장합니다.[3][2][1]

## 평가와 시각화
- 양자화 오차(QE): 평균 $$\|x - w_{BMU(x)}\|$$. 작을수록 코드북 근사가 좋습니다 [1].  
- 토포그래픽 에러(TE): BMU와 차최근 노드의 격자 비인접 비율. 위상 보존도를 가늠합니다.[1]
- U-Matrix: 인접 노드 가중치 간 거리 맵. 큰 값은 군집 경계를 암시합니다.[1]

## 확장과 변형
- 승자빈도 보정: 특정 노드 과점 현상을 줄여 균형을 맞춥니다.[11]
- 고차원 유사도: 코사인/상관거리로 BMU 탐색을 바꿔 텍스트·스펙트럼에 적응합니다.[12]
- 대안 모형: 확률적 해석·수렴성이 중요하면 GTM(EM·RBF 맵핑)을 고려합니다.[8][9]

## 응용 사례:
SOM은 고차원 데이터의 시각적 군집 탐색, 이상치 탐지, 코드북 기반 압축/검색에 쓰입니다. 최근에는 전통 SOM을 딥러닝 파이프라인과 결합하는 하이브리드 적용도 보고됩니다(예: 포즈 추정의 특징 집약 전처리).[13][14]

## 실험 체크리스트
- 전처리: 표준화/정규화 후 학습합니다.[1]
- 맵 크기: 데이터 복잡도 대비 충분히 크게, 과대설정보다는 점진적 확대를 권장합니다.[1]
- 스케줄: $$\sigma_0 \approx \max(m,n)/2$$, $$\alpha_0 \in [0.1,0.5]$$에서 시작해 감쇠합니다.[3][2]
- 검증: QE·TE·U-Matrix로 수렴과 위상 보존을 함께 모니터링합니다.[1]

## 더 공부하면 좋은 자료
- Kohonen의 알고리즘 요약과 툴박스 설명은 이웃 함수·스케줄 설계를 명확히 정리합니다.[2]
- GTM 원전은 SOM의 확률적 관점과 EM 학습을 이해하는 데 도움됩니다.[9][8]
- 교육 자료는 SOM–VQ 관계, 생물학적 배경, 품질 지표의 직관을 잘 정리합니다.[3][1]

[1](https://wiki.ubc.ca/Course:CPSC_522/Self-Organizing_Maps)
[2](http://www.cis.hut.fi/somtoolbox/theory/somalgorithm.shtml)
[3](https://coursepages2.tuni.fi/tiets07/wp-content/uploads/sites/110/2019/01/Neurocomputing3.pdf)
[4](https://arxiv.org/html/2404.00016v1)
[5](https://www.philadelphia.edu.jo/academics/qhamarsheh/uploads/Lecture%2015_Self-Organizing%20Maps%20(Kohonen%20Maps).pdf)
[6](https://www.sciencedirect.com/science/article/abs/pii/S089360801100253X)
[7](https://www.research.ed.ac.uk/en/publications/developments-of-the-generative-topographic-mapping)
[8](https://en.wikipedia.org/wiki/Generative_topographic_map)
[9](https://www.microsoft.com/en-us/research/wp-content/uploads/1998/01/bishop-gtm-ncomp-98.pdf)
[10](https://www.sciencedirect.com/science/article/abs/pii/S0925231298000435)
[11](http://ieeexplore.ieee.org/document/6514307/)
[12](http://ijeecs.iaescore.com/index.php/IJEECS/article/view/26076)
[13](https://www.mdpi.com/2218-6581/13/8/114)
[14](http://ieeexplore.ieee.org/document/537105/)
[15](https://en.wikipedia.org/wiki/Self-organizing_map)
[16](https://linkinghub.elsevier.com/retrieve/pii/S0893608005900002)
[17](https://linkinghub.elsevier.com/retrieve/pii/S0893608009800014)
[18](https://www.semanticscholar.org/paper/7bd2bb8319a75d9140fd4c30431c7283a6b25710)
[19](https://www.semanticscholar.org/paper/a9996df83ed564306046279a0cd4d0cf77450fd7)
[20](https://www.semanticscholar.org/paper/5d1aa1e4f002c71549726cbb8aae9a9b05dac293)
[21](https://link.springer.com/10.1007/s11042-021-10912-1)
[22](http://arxiv.org/pdf/2504.03584.pdf)
[23](https://arxiv.org/pdf/2104.13971.pdf)
[24](https://arxiv.org/abs/2302.07950)
[25](http://arxiv.org/pdf/2406.03832.pdf)
[26](https://arxiv.org/pdf/2109.11769.pdf)
[27](https://arxiv.org/abs/0709.3461)
[28](http://arxiv.org/pdf/2404.00016.pdf)
[29](https://arxiv.org/pdf/1208.1819.pdf)
[30](https://onlinelibrary.wiley.com/doi/10.1002/cam4.217)
[31](https://www.maxwellsci.com/announce/RJASET/15-190-196.pdf)
[32](https://en.wikipedia.org/wiki/Teuvo_Kohonen)
[33](https://eklavyafcb.github.io/docs/KohonenThesis.pdf)
[34](https://www.wikidata.org/wiki/Q1136838)
[35](https://cran.r-project.org/web/packages/kohonen/kohonen.pdf)
[36](http://eprints.aston.ac.uk/1245/1/NCRG_98_024.pdf)
[37](https://dl.acm.org/doi/abs/10.1162/089976698300017953)
[38](https://www.semanticscholar.org/paper/GTM:-The-Generative-Topographic-Mapping-Bishop-Svens%C3%A9n/2639515c248f220c73d44688c0097a99b01e1474)
[39](http://publications.aston.ac.uk/1128/)

https://en.wikipedia.org/wiki/Self-organizing_map
