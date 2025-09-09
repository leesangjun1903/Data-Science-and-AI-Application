# Mutual Information(MI) and KL-Divergence 가이드

Information Theory에서 Entropy는 measure 즉 정보량을 측정하는 도구로 "Entropy가 높다는 것 = Uncertainty가 높다는 것 = Information 양이 많다는 것"을 의미합니다.

이런 배경속에서 Mutual Information(MI)는 2개의 R.V.(Random Variable)들 간의 상호의존성(mutual dependence)을 확인하는 지표입니다. 다른 말로 Information Gain, KL-Divergence이라고도 알려져 있습니다. 정확히는 KL-Divergence와는 다릅니다.

다음 글은 Mutual Information(MI)를 직관부터 수식, 성질, KL-Divergence와의 관계, 그리고 실전 코드 예제로 이어지는 가이드입니다. 핵심은 MI가 두 확률변수의 **의존성**을 정량화하고, KL로 깔끔히 표현되며, 딥러닝에서는 대조학습(InfoNCE)·JSD 등 변분 경계로 추정하고 최대화한다는 점입니다.[1][2][3]

## 왜 MI를 배우나
- MI는 두 변수 간 공유 정보량을 측정합니다. 독립이면 MI=0이고, 완전 결정적 관계면 MI는 한쪽 엔트로피와 같아집니다.[4][2]
- 표현학습과 특징선택에서 유용합니다. 지도학습에선 유용한 피처 선별에, 자가/대조학습에선 정보 보존적 표현을 학습하는 데 쓰입니다.[5][3]

## 정의와 직관
- 정의: MI는 “한 변수를 알 때 다른 변수의 불확실성이 얼마나 줄어드는가”입니다.[2][4]
- 직관: 독립이면 조건분포와 주변분포가 같아 정보 이득이 0이 됩니다. 반대로 한 변수가 다른 변수의 함수이면 정보가 완전히 겹칩니다.[4][2]

## 수식(이산·연속)
- 이산형: $$I(X;Y)=\sum_{x}\sum_{y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)} $$[1][4]
- 연속형: $$I(X;Y)=\int\int p(x,y)\log\frac{p(x,y)}{p(x)p(y)}dxdy $$[1][4]

```math
[p(x,y) = p(x)p(y)
]
```

이므로,

```math
[I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x)p(y)}{p(x)p(y)} = \sum_{x,y} p(x,y) \log 1 = 0
]
```

## KL-Divergence와의 관계
- 핵심 등식: $$I(X;Y)=D_{KL}\big(p_{XY}\|p_Xp_Y\big) $$ [1][2].  
- 또 다른 표현: $$I(X;Y)=\mathbb{E}\_{Y}\left[D_{KL}\big(p_{X|Y}\|p_X\big)\right]=\mathbb{E}\_{X}\left[D_{KL}\big(p_{Y|X}\|p_Y\big)\right] $$ [1][4].  

MI(Mutual Information)와 KL-Divergence(Kullback-Leibler Divergence)는 긴밀하게 연결된 개념입니다.  
Mutual Information은 두 확률 변수 간의 의존성 정도를 나타내며, 이는 두 변수의 공동 분포 (P(x,y))와 독립일 경우의 분포 (P(x)P(y)) 사이의 KL-Divergence로 정의됩니다. 

### KL-Divergence (Kullback–Leibler divergence)
**KL-Divergence (Kullback–Leibler divergence)** 는 두 확률 분포 간의 차이를 정량적으로 측정하는 비대칭적 통계적 거리 척도입니다. 주된 역할은 참 분포 (P)를 모델링하는 분포 (Q)가 실제 분포와 얼마나 다른지 평가하는 것입니다.

수학적으로, 이산 확률 변수에 대해 KL 다이버전스는 다음과 같이 정의됩니다:

```math
[D_{\mathrm{KL}}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}
]
```

여기서 (P)는 실제 분포, (Q)는 비교 대상 분포입니다. 이 값은 항상 0 이상이며, 0일 때 두 분포가 완전히 같음을 의미합니다. 그러나 대칭성을 지니지 않아 $(D_{\mathrm{KL}}(P \parallel Q) \neq D_{\mathrm{KL}}(Q \parallel P))$ 입니다.

주요 특징 및 활용은 다음과 같습니다:

- 정보 이론적 해석: (Q)를 사용해 (P)를 모델링할 때 발생하는 기대 초과 서프라이즈(예상치 못한 정보량)를 나타냄.

- 비대칭성: 거리(metric)라기보다는 ‘발산’(divergence)으로, 삼각부등식도 만족하지 않음.

- 적용 분야: 머신러닝(예: 손실 함수, 분포 비교), 신경망 최적화, 데이터 분포 변화 감지, 생성 모델(GANs) 등에 폭넓게 쓰임.

- 계산 조건: (P)의 확률이 0이 아니면 반드시 (Q)도 0이 아니어야 정의 가능(절대 연속성 조건).

요약하자면, KL 다이버전스는 두 확률 분포의 비슷함 정도를 수치화하는 정보이론 기반 도구로, 특히 모델 성능 평가나 분포 간 차이를 설명하는 데 필수적입니다.

## 기본 성질
- 비음수성: $$I(X;Y)\ge 0$$, 같음 성립은 독립일 때뿐입니다(젠슨 부등식·Gibbs 불평등에서 유도).[2][1]
- 대칭성: $$I(X;Y)=I(Y;X)$$ 이고, 자기정보 $$I(X;X)=H(X)$$입니다.[4][2]

## 엔트로피 분해식
- 합동표현: $$I(X;Y)=H(X)+H(Y)-H(X,Y) $$.[1][4]
- 조건엔트로피 표현: $$I(X;Y)=H(X)-H(X|Y)=H(Y)-H(Y|X) $$ [1][4].

### $$I(X;Y)=H(X)+H(Y)-H(X,Y) $$
두 변수 (X)와 (Y) 각각의 엔트로피(불확실성) 합에서 두 변수를 동시에 고려한 결합 엔트로피 ( H(X,Y) )를 빼줌으로써, 두 변수 사이의 중복되거나 공유되는 정보량(즉, 상호 의존성)을 정확히 나타내기 위함입니다.

## 조건부 MI와 확장
- 조건부 MI: $$I(X;Y|Z)=\mathbb{E}_{Z}\big[I(X;Y)\text{ under }p(\cdot|Z)\big] $$ 로 정의되며, 삼각부등식 형태의 정보 동등식들을 이룹니다[1][2].  
- 정규화 MI(NMI)는 비교를 위한 스케일링 버전입니다. 구현·정의는 문맥에 따라 다릅니다.[6][2]

## 딥러닝에서의 MI 추정과 최대화
- 직접 추정은 어려워 변분 경계 사용이 일반적입니다. 대표적으로 Donsker–Varadhan(DV), InfoNCE, JSD 경계가 쓰입니다.[3][7]
- InfoNCE는 강력하지만 많은 negative가 필요합니다. JSD 경계는 batch 크기 민감도가 낮다는 보고가 있습니다.[8][3]

### Donsker–Varadhan(DV)
Donsker–Varadhan 표현은 KL 발산(KL divergence)을 변분(lower bound) 형태로 나타내는 수식입니다. 주어진 두 확률분포 (P)와 (Q)에 대해 다음과 같이 표현됩니다.

```math
[D_{KL}(P | Q) = \sup_{T: \Omega \to \mathbb{R}} \mathbb{E}_P[T] - \log \mathbb{E}_Q[e^T]
]
```

여기서 (T)는 임의의 실수 값을 갖는 함수이며, ($\sup$)는 상계(최댓값)를 뜻합니다. 이 표현은 직접 (P, Q)를 알기 어려운 상황에서 KL 발산 값을 추정하거나 최적화할 때 유용하며, 신경망을 통해 함수 (T)를 근사하는 방식 등으로 활용됩니다.

특별히 $( T^{*}(x) = \log \frac{P(x)}{Q(x)} )$일 때, 등식이 성립합니다. 즉,

```math
[D_{\mathrm{KL}}(P | Q) = \mathbb{E}_{P} \left[ \log \frac{P(X)}{Q(X)} \right]
]
,

(
[\mathbb{E}_Q \left[\frac{P(x)}{Q(x)}\right] = \int \frac{P(x)}{Q(x)} Q(x) , dx = \int P(x) , dx = 1
]
)
```

는 기존 정의와 동일합니다.

### InfoNCE(Information Noise-Contrastive Estimation)
InfoNCE(Information Noise-Contrastive Estimation) 손실 함수는 대조학습(contrastive learning)에서 많이 쓰이며, 긍정 샘플과 음성 샘플 간의 구분을 최대화하도록 학습하는 함수입니다. 주요 목적은 데이터 포인트 간 의미 있는 관계를 학습하는 것입니다.

InfoNCE의 수식은 다음과 같이 표현할 수 있습니다:

```math
[\mathcal{L}_{\text{InfoNCE}} = - \log \frac{\exp(f_\theta(x, y))}{\exp(f_\theta(x, y)) + \sum_{\tilde{y}} \exp(f_\theta(x, \tilde{y}))}
]
```

여기서

$(f_\theta(x,y))$는 입력 (x)와 긍정 샘플 (y) 간의 유사도 점수 (또는 affinity function)이며,
$(\tilde{y})$는 음성 샘플들을 의미합니다.
즉, 분자는 긍정 샘플과의 유사도를, 분모는 모든 음성 샘플과 긍정 샘플을 포함한 유사도들의 합을 나타내 의미 있는 표현 학습을 유도합니다.

간단히 말해, InfoNCE는 긍정 샘플과 음성 샘플을 확률적으로 구분하는 cross-entropy와 유사하며, 이를 최대화하면 모델이 데이터 간 상호 정보(mutual information)를 최대로 학습하게 됩니다.

#### affinity function
데이터 처리/머신러닝에서의 affinity function은 두 항목 간의 유사도 또는 친밀도를 측정하는 함수로, 예를 들어 거래 데이터 내 항목 간의 관련성(affinity)을 나타내는 데 쓰입니다. 이는 흔히 지지도(support)를 이용해 계산하며, Jaccard 유사도와 같거나 유사한 형태로 정의되기도 합니다. 예를 들어, 항목 (i, j) 사이의 affinity는

```math
[A(i,j) = \frac{supp({i,j})}{supp({i}) + supp({j}) - supp({i,j})}
]
```

로 표현되며, 이는 Aggarwal 등이 정의한 방식입니다.

##### Jaccard Similarity
Jaccard Similarity는 두 집합 간의 유사도를 측정하는 지표로, 두 집합의 교집합 크기를 합집합 크기로 나눈 값입니다. 값의 범위는 0에서 1 사이이며, 1에 가까울수록 두 집합이 더 유사함을 의미합니다.

수식으로는 다음과 같습니다:

```math
[J(A, B) = \frac{|A \cap B|}{|A \cup B|}
]
```

0일 때는 두 집합이 완전히 다름을 뜻하고,
1일 때는 완전히 동일함을 뜻합니다.
주로 텍스트 유사도, 추천 시스템, 이미지 분석 등 다양한 분야에서 활용됩니다. 예를 들어, 두 문서의 단어 집합을 비교해 유사도를 계산하는 데 쓰입니다.

### Jensen-Shannon Divergence (JSD, JS Divergence)
Jensen-Shannon Divergence (JS Divergence)는 두 확률 분포 간의 유사성을 측정하는 대칭적 거리 척도입니다. Kullback-Leibler 발산을 기반으로 하지만, JS Divergence는 항상 유한하고 대칭적이라는 점에서 차이가 있습니다.

구체적으로, JS Divergence는 두 분포 $(P_a)$와 $(P_d)$에 대해 중간 분포 $(P = \frac{1}{2}(P_a + P_d))$를 정의하고, 다음 식으로 계산됩니다:

```math
[JS = \frac{1}{2} KL(P_a || P) + \frac{1}{2} KL(P_d || P)
]
```

여기서 $(KL(\cdot || \cdot))$ 은 Kullback-Leibler 발산(KL Divergence)입니다.

값 범위는 0부터 $(\ln(2))$ 사이이며, 0에 가까울수록 두 분포가 비슷함을 뜻합니다.
JS Divergence의 제곱근은 Jensen-Shannon 거리로 쓰이며, 이는 거리(metric)의 성질을 만족합니다.
이 지표는 머신러닝, 생물정보학, 사회과학 등 여러 분야에서 분포 간 차이 분석에 활용됩니다.  
Python SciPy 라이브러리에서도 scipy.spatial.distance.jensenshannon 함수로 쉽게 계산할 수 있습니다. 최근에는 고밀도 분포에 적합한 변형 논문도 발표되었습니다.

## 실전1: scikit-learn으로 특징선택
- mutual_info_classif는 kNN 기반 비모수 추정으로 각 피처와 라벨 간 MI 점수를 반환합니다.[9][5]
- 고차원 분류에서 상위 K개 유익한 피처를 고르는 데 실무적으로 간편합니다.[10][5]

코드 예시:
```python
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1) 데이터 생성
X, y = make_classification(n_samples=2000, n_features=50, n_informative=10,
                           n_redundant=10, random_state=42)

# 2) MI 점수 계산
mi = mutual_info_classif(X, y, n_neighbors=5, random_state=42)
print("MI scores (first 10):", mi[:10])

# 3) 상위 15개 피처 선택 + 로지스틱 회귀 파이프라인
pipe = Pipeline([
    ("select", SelectKBest(mutual_info_classif, k=15)),
    ("scale", StandardScaler(with_mean=False)),  # 희소 행렬 대비 일반성
    ("clf", LogisticRegression(max_iter=200))
])
pipe.fit(X, y)
print("Pipeline trained with MI-based feature selection.")
```
- mutual_info_classif는 비음수 MI를 반환합니다. discrete/continuous 처리와 kNN 근방 수를 선택적으로 조정합니다.[5]
- SelectKBest와 결합해 간단히 차원 축소와 학습 파이프라인을 구성할 수 있습니다.[10][9]

## 실전2: PyTorch로 MI 최대화(InfoNCE 스타일)
- 목표: 인코더 $$f_\theta$$의 표현 $$z=f_\theta(x)$$가 입력과 높은 **상호정보량**을 갖도록 학습합니다.[7][3]
- InfoNCE 손실은 긍정쌍 대비 부정쌍 분별을 통해 MI 하한을 최대화합니다. 단, 많은 negative가 필요할 수 있습니다.[11][3]

코드 스케치:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, d)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.net(x)
        return F.normalize(z, dim=1)

def info_nce_loss(z_q, z_k, temperature=0.2):
    # z_q: queries, z_k: keys (긍정쌍은 같은 이미지의 변환)
    logits = z_q @ z_k.t() / temperature                 # BxB 유사도
    labels = torch.arange(z_q.size(0), device=z_q.device) # 대각 성분이 positive
    return F.cross_entropy(logits, labels)

# 예시 학습 루프(약식)
encoder = Encoder()
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

for x in dataloader:  # x: Bx1x28x28 같은 이미지 배치
    x1 = augment(x); x2 = augment(x)     # 두 가지 stoch. 변환
    z1 = encoder(x1)
    z2 = encoder(x2)
    loss = (info_nce_loss(z1, z2) + info_nce_loss(z2, z1)) * 0.5
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
- InfoNCE는 $$\mathrm{O}(B)$$ negative를 활용하며, 지역/전역 특성 확장 시 복잡도가 커질 수 있습니다.[3]
- 대안으로 JSD 기반 경계는 negative에 덜 민감하며, 대규모 아키텍처에 실용적일 수 있습니다.[8][3]

## 실전3: PyTorch로 JSD 기반 MI 최대화(Deep InfoMax 스타일)
- 아이디어: discriminator $$T_\omega$$가 긍정쌍(global-local, global-global)과 부정쌍을 구분하도록 학습하여 JSD 경계를 최대화합니다.[8][3]
- 장점: negative 수에 덜 민감해 배치 크기 제약이 완화됩니다.[3][8]

코드 스케치:
```python
class Discriminator(nn.Module):
    def __init__(self, d_local, d_global, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_local + d_global, h), nn.ReLU(),
            nn.Linear(h, 1)
        )
    def forward(self, l, g):  # l: local feat (B*M*M, d_l), g: global feat (B, d_g)
        if g.dim() == 2 and l.dim() == 2:
            g = g.repeat_interleave(l.size(0)//g.size(0), dim=0)
        x = torch.cat([l, g], dim=1)
        return self.net(x)

def jsd_loss(T_pos, T_neg):
    # JSD-based MI lower bound surrogate
    # E_pos[softplus(-T)] + E_neg[softplus(T)]  (binary cross-entropy 형태)
    return F.softplus(-T_pos).mean() + F.softplus(T_neg).mean()

# 전역/지역 특징 뽑기 예시(간단화를 위해 MLP로 대체)
encoder = Encoder(d=128)
disc = Discriminator(d_local=128, d_global=128)
opt = torch.optim.Adam(list(encoder.parameters())+list(disc.parameters()), lr=1e-3)

for x in dataloader:
    z_global = encoder(augment(x))               # Bx128
    z_local  = encoder(augment(x))               # Bx128 (데모용; 실제는 CNN의 local map 사용)
    # 긍정: 같은 샘플 쌍
    T_pos = disc(z_local, z_global)
    # 부정: z_global를 셔플
    z_global_neg = z_global[torch.randperm(z_global.size(0))]
    T_neg = disc(z_local, z_global_neg)

    loss = jsd_loss(T_pos, T_neg)
    opt.zero_grad()
    loss.backward()
    opt.step()
```
- 실제 구현은 CNN feature map의 공간 위치별 local feature와 global summary 간 MI를 최대화합니다.[3]
- 전역/지역 목적을 합산하고, prior matching을 추가해 안정화하는 설계도 보고됩니다.[3]

## 이론-실전 연결 포인트
- 등식 $$I(X;Y)=\mathbb{E}_Y[D_{KL}(p_{X|Y}\|p_X)]$$는 “조건이 주어질 때 분포가 얼마나 바뀌는가”의 평균으로 MI를 해석하게 해 줍니다[1].  
- 대조학습 손실(InfoNCE)은 샘플링 기반으로 이 KL 차이를 하한으로 근사하여 최대화한다고 볼 수 있습니다.[11][7]

## 학습 팁
- InfoNCE: temperature 튜닝, 대규모 negative, 데이터 증강 다양성이 중요합니다.[7][3]
- JSD: batch/negative 민감도가 낮아 자원 제약에서 유리하며, 지역·전역 결합으로 성능을 끌어올립니다.[8][3]

## 추가 읽을거리
- Wikipedia: 정의, 성질, KL 표현을 간결히 정리합니다.[4][1]
- Deep InfoMax(ICLR 2019): 지역·전역 MI 최대화 프레임워크와 JSD/InfoNCE 비교를 제공합니다.[7][3]

수식과 구현의 연결을 염두에 두면, MI는 “독립이면 0, 의존성이 커질수록 증가”하는 **의존성 지표**이며, 딥러닝에서는 대조적 목표로 자연스럽게 녹아들어 강건한 표현을 학습하는 도구가 됩니다.[1][3]

[1](https://en.wikipedia.org/wiki/Mutual_information)
[2](http://www.scholarpedia.org/article/Mutual_information)
[3](https://openreview.net/pdf?id=Bklr3j0cKX)
[4](http://home.zcu.cz/~potmesil/ADM%202015/4%20Regrese/Coefficients%20-%20Gamma%20Tau%20etc./Z-Mutual%20information%20-%20Wikipedia.htm)
[5](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)
[6](https://hi-lu.tistory.com/entry/Mutual-Information-%ED%8C%8C%ED%97%A4%EC%B9%98%EA%B8%B0)
[7](https://hui2000ji.github.io/machine_learning/representation_learning_with_mutual_information_maximization/)
[8](https://openaccess.thecvf.com/content/CVPR2023W/TCV/papers/Shrivastava_Estimating_and_Maximizing_Mutual_Information_for_Knowledge_Distillation_CVPRW_2023_paper.pdf)
[9](https://scikit-learn.org/stable/modules/feature_selection.html)
[10](https://sklearner.com/scikit-learn-mutual_info_classif/)
[11](https://arxiv.org/abs/2308.15704)
[12](https://aigong.tistory.com/43)
[13](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
[14](https://deep-learning-study.tistory.com/768)
[15](https://www.quantiki.org/wiki/mutual-information)
[16](https://static.hlt.bme.hu/semantics/external/pages/pointwise_mutual_information/en.wikipedia.org/wiki/Mutual_information.html)
[17](https://data-newbie.tistory.com/430)
[18](https://pages.stern.nyu.edu/~dbackus/BCZ/entropy/Mutual-information-Wikipedia.pdf)
[19](https://www.sciencedirect.com/science/article/abs/pii/S0016003223005458)
[20](https://chealin93.tistory.com/274)
[21](https://stackoverflow.com/questions/70820610/discrete-features-parameter-in-sklearn-feature-selection-mutual-info-classif)

# Reference
https://aigong.tistory.com/43
