# Von Mises–Fisher(vMF) distribution
폰 미제서 피셔 분포는 쉽게 말해 (p-1)차원상의 확률분포, 예를들어 3차원상의 가우시안 분포 등 을 말합니다.

![image](https://github.com/user-attachments/assets/3465741b-700d-4cb6-a7f5-a6a06938eee0)

vMF는 단위 구면 위 방향 데이터의 확률 모델로, 평균 방향과 집중도를 통해 “벡터의 방향성”을 정규분포처럼 다루는 **구면 분포**입니다.[1][2]

## 핵심 요약
- vMF는 단위 구면 $$\mathbb{S}^{d-1} $$ 위의 확률분포로, 평균 방향 $$\boldsymbol{\mu} $$와 집중도 $$\kappa $$로 정의됩니다.[2][1]
- 확률밀도는 $$p(\boldsymbol{x}\mid \boldsymbol{\mu},\kappa)=C_d(\kappa)\exp(\kappa \boldsymbol{\mu}^\top \boldsymbol{x}) $$이며, 정규화 상수는 수정 베셀함수 $$I_\nu$$를 포함합니다.[3][1]
- $$\kappa\to 0 $$이면 균일분포에 수렴하고, $$\kappa $$가 클수록 **집중**이 강해집니다.[3][1]

### 기호 설명:
- $(\boldsymbol{x})$: 확률변수로, 단위 벡터 (길이가 1인 벡터)이며, 차원은 (d)입니다. : 관찰되는 방향 데이터 (차원 (d)의 구면 위의 점)
- $(\boldsymbol{\mu})$: 평균 방향 벡터 (mean direction vector)로, 단위 벡터이며 $(|\boldsymbol{\mu}|=1)$을 만족합니다. : 데이터가 집중되는 중심 방향
- $(\kappa)$: 집중도(concentration) 파라미터로, $(\kappa \geq 0)$이고 값이 클수록 $(\boldsymbol{x})$가 $(\boldsymbol{\mu})$ 주변에 더 몰림을 의미합니다.
- $(C_d(\kappa))$: 정규화 상수(normalizing constant)로, 확률밀도가 1이 되도록 하는 상수입니다. : 확률밀도 함수가 구면에서 적분하여 1이 되도록 조절하는 역할입니다.

다음과 같이 표현됩니다.
```math
[ C_d(\kappa) = \frac{\kappa^{d/2 - 1}}{(2\pi)^{d/2} I_{d/2 - 1}(\kappa)}
]
```

여기서 $(I_{v}(\kappa))$는 수정 베셀 함수(modified Bessel function of the first kind)입니다.  
요약하자면, vMF 분포는 (d)-차원 단위 구면 위의 분포로, $(\boldsymbol{\mu})$ 방향을 중심으로 $(\kappa)$ 값에 의해 밀도가 조절되는 분포입니다.

#### modified Bessel function of the first kind
VMF 분포(von Mises-Fisher 분포)에서 사용하는 **수정 베셀 함수(modified Bessel function of the first kind, $I_\nu(x)))$** 는 다음과 같이 정의되고 특징지어집니다.


$(I_\nu(x))$ 는 수정 베셀 미분 방정식의 해이며, 일반 베셀 함수 $(J_\nu(x))$ 와 복소수 변환을 통해 관련됩니다:

```math
[ I_\nu(x) = i^{-\nu} J_\nu(i x)
]
```
여기서 (i)는 허수 단위입니다.

이 함수는 진동하지 않고, 지수 함수처럼 증가하거나 감쇠하는 성질을 갖습니다. VMF 분포에서는 벡터 유사성을 측정하는 데 사용되며, 분포의 밀도 함수 계산에 필수적입니다.

수식적으로는 적분 표현과 급수 전개 등 다양한 형태로 나타낼 수 있으며, 예를 들어 다음과 같이 적분식으로 표현 가능합니다:

```math
[ I_n(x) = \frac{1}{\pi} \int_0^\pi e^{x \cos \theta} \cos(n \theta) d\theta
]
```

$((n=0,1,2,\dots)$ 일 때).

VMF 분포에서 이 함수는 분포의 정규화 상수(normalizing constant) 계산에 중요한 역할을 하며, 높은 차원에서 계산적 난이도가 있으므로 수치적 방법이 자주 사용됩니다.

요약하면, VMF 분포에서 사용되는 수정 베셀 함수는 $(\mathbf{I}_\nu(x))$ 로 표기되며, 베셀 함수의 복소수 확장으로 정의되고, 분포의 밀도 함수 정규화에 핵심적으로 기여하는 특별 함수입니다.

수학적으로, 변형 베셀 함수 $(I_{\nu}(x))$ 는 다음과 같은 변형 베셀 방정식의 해입니다:

```math
[x^2 y'' + x y' - (x^2 + \nu^2) y = 0
]
```

이 함수는 진동하지 않고, 입력값이 커질수록 지수적으로 증가합니다. von Mises 분포에서 제0종 변형 베셀 함수 $(I_0(x))$ 가 정규화 상수를 보장하는 역할을 합니다.

요약하면, vmf 분포의 핵심 수식에 포함되는 제1종 변형 베셀 함수는 분포의 확률 밀도 함수가 올바른 확률 분포가 되도록 정규화하는 데 반드시 필요한 수학적 함수이며, 이는 진동하지 않고 지수적인 특성을 가진다는 점에서 일반 베셀 함수와 구분됩니다.

##### 왜 그런가?
von Mises 분포에서 정규화 상수는 수정 베셀 함수 제0종 $(I_0(\kappa))$ 로 표현되며, 이 함수가 포함된 분포 밀도 함수의 적분값(또는 합계)이 1이 되도록 보장합니다.

von Mises 분포 밀도 함수는 다음과 같습니다:

```math
[f(\theta|\mu, \kappa) = \frac{e^{\kappa \cos(\theta - \mu)}}{2\pi I_0(\kappa)},
]
```

여기서 $(\kappa)$는 집중도, $(\mu)$는 평균방향입니다.

정규화 상수 역할: $(I_0(\kappa))$는 수정 베셀 함수 제0종이며, 0에서 ∞까지가 아닌 $([0, 2\pi])$ 구간에서 함수 $(e^{\kappa \cos \phi})$ 의 적분값과 관련있습니다.

- 총합(적분)이 1이 되는 이유: 확률밀도함수가 정상적인 분포가 되려면 다음 적분 조건이 성립해야 합니다.

```math
[\int_0^{2\pi} f(\theta|\mu, \kappa) d\theta = 1.
]
```

이를 $(f(\theta))$ 식에 대입하여 적분하면,

```math
[\int_0^{2\pi} \frac{e^{\kappa \cos(\theta-\mu)}}{2\pi I_0(\kappa)} d\theta = \frac{1}{2\pi I_0(\kappa)} \int_0^{2\pi} e^{\kappa \cos \phi} d\phi = 1,
]
```

여기서 치환: $(\phi = \theta - \mu)$.

중요한 식 (정규화 상수를 정의하는 적분):

```math
[I_0(\kappa) = \frac{1}{2\pi} \int_0^{2\pi} e^{\kappa \cos \phi} d\phi,
]
```

따라서,

```math
[\int_0^{2\pi} e^{\kappa \cos \phi} d\phi = 2 \pi I_0(\kappa).
]
```

이로써 분모에 $(2 \pi I_0(\kappa))$를 둠으로써 밀도 함수의 적분값이 1이 되어 정규화가 보장됩니다.

즉, 수정 베셀 함수 $(I_0(\kappa))$ 는 von Mises 분포의 분포 함수가 전체 구간에서 확률합 1이 되게 하는 정규화 상수 역할을 수행합니다.

1차원 von Mises 분포 식:

```math
[ f(\theta|\mu, \kappa) = \frac{e^{\kappa \cos(\theta - \mu)}}{2\pi I_0(\kappa)}
]
```

여기서 $(\theta, \mu \in [0, 2\pi))$, $(\kappa)$는 집중도, $(I_0)$는 제1종 변형 베셀 함수.

다차원 von Mises–Fisher 분포 식:

```math
[ p(\boldsymbol{x}|\boldsymbol{\mu}, \kappa) = C_d(\kappa) \exp(\kappa \boldsymbol{\mu}^\top \boldsymbol{x})
]
```

여기서 $(\boldsymbol{x})$는 단위 벡터, $(\boldsymbol{\mu})$ 는 평균 방향, $(C_d(\kappa))$는 정규화 상수.

1차원 원에서는 벡터 $(\boldsymbol{x} = (\cos\theta, \sin\theta))$ 와 $(\boldsymbol{\mu} = (\cos\mu, \sin\mu))$ 로 표현 가능하며, 내적 $(\boldsymbol{\mu}^\top \boldsymbol{x} = \cos(\theta - \mu))$ 가 되어 두 식이 일치하게 변환됩니다. 정규화 상수 $(C_d(\kappa))$는 $(1/(2\pi I_0(\kappa)))$와 대응합니다.

따라서, von Mises 분포는 1차원 원 위의 확률 분포이고, von Mises-Fisher 분포는 이를 고차원 구면에 확장한 형태입니다.

## 왜 vMF인가
vMF는 “방향” 그 자체를 모델링합니다. 임베딩을 $$L^2$$-정규화해 코사인 유사도 체계로 학습하는 환경에서, 데이터의 조건부 분포를 vMF로 가정하면 손실과 추정이 수학적으로 깔끔해집니다. 예를 들어 구면 클러스터링(텍스트/이미지 임베딩)이나 의료영상의 방향 필드 같은 문제에서 자연스럽게 적용됩니다. 또한 등방성 가우시안을 단위구면에 “제한(restriction)”하면 vMF가 되므로, 정규분포의 구면 아날로그로 이해할 수 있습니다.[4][5][6][7][8][1]

## 수학 정리
- 정의: $$\boldsymbol{x}\in \mathbb{S}^{d-1} $$, $$\boldsymbol{\mu}\in \mathbb{S}^{d-1} $$, $$\kappa\ge 0 $$일 때  
  $$p(\boldsymbol{x}\mid \boldsymbol{\mu},\kappa)=C_d(\kappa)\exp\big(\kappa\,\boldsymbol{\mu}^\top \boldsymbol{x}\big) $$[1][3]
- 정규화 상수:  
  $$C_d(\kappa)=\frac{\kappa^{d/2-1}}{(2\pi)^{d/2}I_{d/2-1}(\kappa)} $$[3][1]
- 평균 및 일치 통계: $$\mathbb{E}[\boldsymbol{X}]=A_d(\kappa)\boldsymbol{\mu} $$, 여기서 $$A_d(\kappa)=\frac{I_{d/2}(\kappa)}{I_{d/2-1}(\kappa)} $$이며, 최대우도추정에서 $$\hat{\boldsymbol{\mu}}$$는 표본 평균방향이고 $$\hat{\kappa}$$는 $$A_d(\kappa)$$의 역함수로 구합니다(근사/뉴턴법 사용).[5][4]
- 가우시안과의 관계: 등방성 가우시안을 단위구면에 제한하면 vMF가 됩니다. 따라서 “정규분포의 구면 버전”으로 이해됩니다.[7][1]

## 계산 이슈와 팁
vMF의 난점은 $$I_\nu(\kappa)$$와 그 비율 $$A_d(\kappa)$$의 수치적 안정성입니다. 고차원/대 $$\kappa$$에서 언더/오버플로가 쉽게 발생합니다. 연구용 구현은 고정소수/임의정밀도를 허용하는 특수함수 라이브러리나 안전한 근사식을 활용합니다. PyTorch 구현에서는 mpmath를 통한 고정밀 보조, 로그-정규화 기법, 큰 $$\kappa$$ 근사(예: $$A_d(\kappa)\approx 1-\frac{d-1}{2\kappa}$$) 등을 조합합니다.[4][5]

## 샘플링
샘플링은 “평균 방향의 축 정렬 → 스칼라 코사인 성분 샘플 → 무작위 균일 회전 복원” 절차를 사용합니다. 효율적 방법으로 Wood(1994) 계열의 알고리즘이 널리 쓰이며, R/pytorch 구현 레퍼런스가 공개되어 있습니다. 실전에서는 좌표축 정렬과 하우스홀더 반사를 함께 사용하면 간단하고 안정적입니다.[9][10]

### 평균 방향의 축 정렬
샘플링을 위해 주어진 평균 방향 벡터 $(\mu)$를 기준 좌표축에 정렬하는 과정입니다.  
주로 기준 축 (예: z축)과 $(\mu)$ 사이의 회전을 계산해 벡터들을 축 방향에 맞추어 단순화합니다. 이후 스칼라 성분 샘플링을 편리하게 할 수 있습니다.

### 스칼라 코사인 성분 샘플
vmf 분포는 구면 위의 확률 분포로, 방향 벡터의 평균 방향과의 코사인 유사도(즉, 내적값)에 해당하는 스칼라 성분을 샘플링하는 단계입니다. Wood(1994) 알고리즘이 이 작업을 효율적으로 수행하는 대표적 방법이며, 해당 코사인 성분을 뽑아내면 나중에 무작위 회전을 통해 원래 공간으로 복원합니다.

#### Wood(1994)
Wood(1994)의 von Mises-Fisher (vMF) 분포 샘플링 알고리즘은, 주로 스칼라 코사인 성분(sample of scalar cosine component)을 생성하는 방법에 대해 거부 샘플링(rejection sampling) 방식을 제안합니다. 이 알고리즘은 vMF 분포의 모달 방향이 (0,...,0,1)일 때, 집중도 파라미터 κ에 기반하여 스칼라 코사인 값을 샘플링합니다.

과정은 다음과 같습니다:

b 계산: $( b = (-2 \kappa + \sqrt{4 \kappa^2 + (d-1)^2})/(d-1) )$, 여기서 ( d )는 차원이고 κ는 집중도 파라미터입니다.

샘플링: 스칼라 값 ( w ) (즉, 스칼라 코사인 성분)를 다음 분포에서 샘플링합니다:

```math
[f(w) \propto (1 - w^2)^{(d-3)/2} \exp(\kappa w), \quad w \in [-1,1]
]
```

이는 거부 샘플링을 사용하여 효율적으로 샘플링합니다.

보조 변수 (b), (x), (c)는 다음과 같이 정의됩니다.

목적: (p)-차원 단위 구면상의 vMF 분포 (\mathrm{vMF}(\mu, \kappa))에서 샘플을 생성하는 것.

보조 변수 정의:

- $(b = \frac{-2\kappa + \sqrt{4\kappa^2 + (p-1)^2}}{p-1})$
- $(x = \frac{1 - b}{1 + b})$
- $(c = \kappa x + (p-1) \ln(1 - x^2))$
여기서 $(\kappa)$는 집중도, (p)는 차원 수입니다.

샘플링 단계:

- (Z)를 베타 분포 $(\mathrm{Beta}\left(\frac{p-1}{2}, \frac{p-1}{2}\right))$에서 샘플링.  
- $(W = \frac{1 - (1+b)Z}{1 - (1-b)Z})$ 계산.
- $(U \sim \mathrm{Uniform}(0,1))$ 샘플링.
- $(\kappa w + (p-1) \ln(1 - x w) - c \geq \ln U)$이면 $(w = W)$를 수용, 아니면 1단계로 돌아감.

- 최종 샘플 생성:

- (w)로부터 구면상의 한 점을 생성하는데, mean 방향 $(\mu)$ 와 직교하는 균일 분포 방향 벡터 (v)를 생성 후 $(x = w \mu + \sqrt{1 - w^2} v)$ 로 최종 표본 생성

Wood(1994)의 이 알고리즘은 rejection sampling에서 제안 분포와 타겟 분포 간 효율적인 균형을 맞추도록 (b, x, c)를 설정하여 높은 수용 확률을 보장합니다.

요약하자면, (b, x, c)는 rejection sampler에서 수용 판정을 위한 함수값 계산에 사용되며, 이 수식들이 Wood가 제안한 핵심 공식입니다.

Wood(1994) 알고리즘의 핵심은 스칼라 코사인 성분 ( w )를 집중도 κ와 차원 d에 기반해 거부 샘플링으로 추출하고, 이를 이용해 vMF 분포상의 점을 생성하는 수학적인 절차에 있습니다.

### 무작위 균일 회전 복원
정렬된 좌표계에서 샘플링한 벡터를 다시 원래 방향 공간으로 변환하는 과정으로, 균일 분포하는 회전(예: 하우스홀더 반사 등)을 적용하여 방향을 원상 복구합니다. 이 단계 덕분에 전체 샘플링이 올바른 vmf 분포를 따르게 됩니다.

### 하우스홀더 반사 (Householder reflection)
주어진 방향 벡터를 다른 축으로 효과적으로 반사시켜 정렬하거나 복원할 때 쓰이는 선형 대수 도구입니다. 기본적으로 벡터를 기준축으로 보내거나 그 역변환을 할 때 수치적 안정성과 효율성을 제공합니다. 실제 구현에서는 평균 방향의 축 정렬과 함께 사용하면 간단하고 견고한 샘플링 절차가 됩니다.

이러한 절차를 적용하면 vmf 분포에서 벡터를 효과적으로 샘플링할 수 있으며, 특히 Wood(1994) 알고리즘이 널리 활용되고, R이나 PyTorch 같은 언어별 구현도 참고할 수 있습니다.

## 연구 활용 예
- 구면 클러스터링/토픽 모델링: Mixture of vMF(MvMF)로 문서/이미지 임베딩을 구면상에서 클러스터링합니다.[11][5]
- 딥러닝 표현 학습: 임베딩을 정규화해 vMF 사후를 가정, 대리(proxy) 기반 학습과 결합하거나 비등방 vMF로 일반화합니다.[6][12]
- 공간/방향 회귀: vMF 오차 모형으로 방향 반응변수에 대한 GLM/베이지안 회귀를 구성합니다.[13][8]

## PyTorch 실전 코드 1: vMF 기본 모듈
확률밀도, 로그정규화, 샘플링, MLE(EM의 M-step)까지 포함한 연구용 스켈레톤입니다. mpmath로 베셀함수와 $$A_d(\kappa)$$의 안정적 계산을 보조합니다.[5][4]

```python
# vmf.py
import torch
import torch.nn.functional as F
from torch import nn
from mpmath import besseli, mp

mp.dps = 80  # high precision for stability [27]

def log_C_d(kappa, d):
    # log normalizer log C_d(kappa) [24][25]
    # C_d = kappa^{d/2-1} / ( (2π)^{d/2} I_{d/2 - 1}(kappa) )
    v = d/2 - 1.0
    k = float(kappa)
    if k == 0.0:
        # uniform on sphere: C_d(0) = 1 / SurfaceArea(S^{d-1})
        # SA = 2 * pi^{d/2} / Gamma(d/2) → logC = -log(SA)
        import mpmath as mpn
        SA = 2.0 * (mpn.pi**(d/2)) / mpn.gamma(d/2)
        return float(-mpn.log(SA))
    Iv = besseli(v, k)
    # logC = (d/2 - 1)*log k - (d/2)*log(2π) - log I_{v}(k)
    logC = (d/2 - 1.0) * mp.log(k) - (d/2) * mp.log(2*mp.pi) - mp.log(Iv)
    return float(logC)  # [24][25]

def A_d(kappa, d):
    # A_d(kappa) = I_{d/2}(kappa) / I_{d/2 - 1}(kappa) [29]
    v = d/2 - 1.0
    k = float(kappa)
    if k == 0.0:
        return 0.0
    num = besseli(v+1.0, k)
    den = besseli(v, k)
    return float(num/den)  # [29]

def invert_A_d(Rbar, d, tol=1e-10, max_iter=100):
    # Solve A_d(kappa)=Rbar for kappa via Newton [29]
    # Good init: for d>=3, kappa ≈ Rbar*(d-1 - Rbar**2)/(1 - Rbar**2) [Sra 2012 style] [29]
    if Rbar < 1e-8:
        return 0.0
    if Rbar > 0.999999:
        # large kappa asymptotics: A ≈ 1 - (d-1)/(2kappa) → kappa ≈ (d-1)/(2*(1-Rbar)) [29]
        return (d-1) / (2*(1 - Rbar) + 1e-12)
    kappa = Rbar * (d - 1 - Rbar**2) / (1 - Rbar**2 + 1e-12)
    for _ in range(max_iter):
        Ad = A_d(kappa, d)
        # derivative: A'(k) = 1 - A(k)^2 - (d-1)/k * A(k) [standard identity] [29]
        Adp = 1 - Ad*Ad - (d-1)/(kappa + 1e-12)*Ad
        step = (Ad - Rbar) / (Adp + 1e-12)
        kappa -= step
        if abs(step) < tol:
            break
        if kappa < 0:
            kappa = 1e-6
    return float(kappa)  # [29]

def householder_rotation(a, b):
    # return orthogonal matrix H s.t. H a = b (both unit) [for sampling] [22]
    a = a / a.norm()
    b = b / b.norm()
    v = (a - b)
    nv = v.norm()
    if nv < 1e-8:
        return torch.eye(a.shape, device=a.device)
    v = v / nv
    H = torch.eye(a.shape, device=a.device) - 2.0 * v.unsqueeze(1) @ v.unsqueeze(0)
    return H  # [22]

def sample_vmf(mu, kappa, n_samples):
    # Wood-style sampling on S^{d-1} [22][26]
    # 1) sample along north pole, 2) random orthogonal, 3) rotate to mu
    d = mu.numel()
    device = mu.device
    # sample w = cos(theta) on [-1,1] with proper density
    # Use rejection sampler for general d [22]
    b = (-2*kappa + torch.sqrt(4*kappa**2 + (d-1.)**2)).item() / (d-1.)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d-1.)*torch.log(torch.tensor(1 - x0**2))
    samples = []
    for _ in range(n_samples):
        while True:
            z = torch.distributions.Beta((d-1)/2., (d-1)/2.).sample().item()
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            u = torch.rand(1).item()
            if kappa*w + (d-1.)*torch.log(torch.tensor(1 - x0*w)) - c >= torch.log(torch.tensor(u)):
                break
        # sample a random unit vector orthogonal part
        v = torch.randn(d, device=device)
        v = v / v.norm()
        orth = v - (v @ torch.tensor([1.0] + [0.0]*(d-1), device=device)) * torch.tensor([1.0] + [0.0]*(d-1), device=device)
        orth = orth / (orth.norm() + 1e-12)
        # construct on north pole basis
        x = torch.zeros(d, device=device)
        x = w
        if d > 1:
            fac = torch.sqrt(torch.clamp(1 - w*w, min=0.0))
            x[1:] = fac * F.normalize(torch.randn(d-1, device=device), dim=0)
        # rotate north pole e1 to mu
        e1 = torch.zeros(d, device=device); e1 = 1.0
        H = householder_rotation(e1, mu)
        samples.append(H @ x)
    return torch.stack(samples, dim=0)  # [22][26]

class VonMisesFisher:
    def __init__(self, mu, kappa):
        self.mu = F.normalize(mu, dim=-1)
        self.kappa = kappa
        self.d = mu.numel()

    def log_prob(self, x):
        x = F.normalize(x, dim=-1)
        logC = log_C_d(self.kappa, self.d)
        return torch.dot(self.mu, x) * self.kappa + torch.tensor(logC, device=x.device)  # [24][25]

    def sample(self, n):
        return sample_vmf(self.mu, torch.tensor(self.kappa), n)  # [22][26]

def vmf_mle(X):
    # X: (N,d), unit vectors; return mu_hat, kappa_hat [29]
    X = F.normalize(X, dim=-1)
    N, d = X.shape
    R = X.sum(dim=0)
    R_norm = R.norm()
    mu_hat = R / (R_norm + 1e-12)
    Rbar = (R_norm / N).item()
    kappa_hat = invert_A_d(Rbar, d)
    return mu_hat, kappa_hat  # [29]
```


- 포인트  
  - log_prob는 로그-정규화 상수를 직접 계산합니다. mpmath로 $$I_\nu$$를 안정적으로 구합니다.[4][3]
  - invert_A_d는 뉴턴법과 대$$\kappa$$ 근사로 $$\kappa$$를 역추정합니다.[5]
  - 샘플러는 Wood 방식 계열을 반영하며, 하우스홀더 반사로 평균 방향으로 회전합니다.[9][10]

## PyTorch 실전 코드 2: vMF 혼합(MvMF) EM 학습
구면 클러스터링을 위한 MvMF의 EM 알고리즘입니다. E-step은 책임도, M-step은 각 컴포넌트별 $$\hat{\mu}_k$$, $$\hat{\kappa}_k$$, $$\hat{\pi}_k$$ 갱신으로 구성됩니다.[11][5]

```python
# vmf_mixture.py
import torch
import torch.nn.functional as F
from vmf import log_C_d, invert_A_d

class VMFMixture:
    def __init__(self, n_components, d, device='cpu'):
        self.K = n_components
        self.d = d
        self.device = device
        self.mu = F.normalize(torch.randn(n_components, d, device=device), dim=-1)
        self.kappa = torch.ones(n_components, device=device) * 10.0
        self.pi = torch.ones(n_components, device=device) / n_components  # [29]

    def log_prob_components(self, X):
        # X: (N,d) unit vectors
        N = X.shape
        # log C_d(kappa)
        logC = torch.tensor([log_C_d(float(k), self.d) for k in self.kappa], device=self.device)  # [24]
        # mu^T x
        dot = X @ self.mu.t()  # (N,K)
        return dot * self.kappa + logC  # (N,K) [24]

    def e_step(self, X):
        # responsibilities r_{nk} ∝ pi_k * f_k(x_n)
        log_fk = self.log_prob_components(X)  # (N,K)
        log_weighted = log_fk + torch.log(self.pi.unsqueeze(0))  # [29]
        log_r = log_weighted - torch.logsumexp(log_weighted, dim=1, keepdim=True)
        r = torch.exp(log_r)
        return r  # (N,K) [29]

    def m_step(self, X, r, min_kappa=1e-3):
        Nk = r.sum(dim=0) + 1e-12  # (K,)
        self.pi = Nk / Nk.sum()  # [29]
        # update mu
        Rk = r.t() @ X  # (K,d)
        mu_new = F.normalize(Rk, dim=-1)
        self.mu = mu_new
        # update kappa via Rbar_k
        Rk_norm = Rk.norm(dim=1)
        Rbar_k = (Rk_norm / (Nk + 1e-12)).cpu().numpy()
        kappa_new = []
        for rb in Rbar_k:
            kappa_new.append(max(invert_A_d(float(rb), self.d), min_kappa))  # [29]
        self.kappa = torch.tensor(kappa_new, device=self.device)

    def fit(self, X, n_iter=100, tol=1e-5):
        X = F.normalize(X, dim=-1)
        prev_ll = -1e30
        for it in range(n_iter):
            r = self.e_step(X)
            self.m_step(X, r)
            ll = self.score(X).item()
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

    def score(self, X):
        log_fk = self.log_prob_components(X)  # (N,K)
        log_weighted = log_fk + torch.log(self.pi.unsqueeze(0))
        ll = torch.logsumexp(log_weighted, dim=1).mean()
        return ll  # [29]
```


- 포인트  
  - E-step: $$r_{nk}\propto \pi_k\,C_d(\kappa_k)\exp(\kappa_k \mu_k^\top x_n)$$를 로그-합-지수로 안정화합니다.[5]
  - M-step: $$\mu_k$$는 가중 평균방향 정규화, $$\kappa_k$$는 $$R_k/N_k$$에서 역함수로 업데이트합니다.[5]

## 딥러닝 결합 아이디어
- 분류/메트릭러닝: 마지막 임베딩을 정규화하고, 클래스별 **vMF likelihood**를 최대화하는 손실을 사용합니다. 이는 코사인 분류기와 등가 형태를 이루며, 집중도 $$\kappa_c$$를 학습 가능 하이퍼로 두면 클래스-내 분산을 제어할 수 있습니다.[6][4]
- 프록시 기반/비등방 일반화: 클래스 프로토타입을 평균 방향으로 두고, 축마다 다른 집중도를 허용하는 변형(nivMF)을 적용하면 표현력이 증가합니다.[6]
- 반지도/세그멘테이션: 픽셀 피처를 구면 임베딩으로 두고, vMF 기반 대비학습·불확실성 제약을 적용해 pseudo-label 오염을 줄이는 사례가 있습니다.[14]

## 실험 스니펫: 합성 데이터 클러스터링
```python
import torch
import torch.nn.functional as F
from vmf_mixture import VMFMixture
from vmf import VonMisesFisher

torch.manual_seed(0)
device = 'cpu'
d = 16
K = 3
N = 600

# 생성용 평균방향과 집중도
true_mu = F.normalize(torch.randn(K, d), dim=-1)
true_kappa = torch.tensor([5.0, 15.0, 30.0])

# 데이터 생성
Xs = []
labs = []
for k in range(K):
    vmf = VonMisesFisher(true_mu[k], float(true_kappa[k]))
    Xk = vmf.sample(N//K)
    Xs.append(Xk)
    labs.append(torch.full((N//K,), k))
X = torch.cat(Xs, dim=0)
y = torch.cat(labs, dim=0)

# MvMF 학습
model = VMFMixture(n_components=K, d=d, device=device)
model.fit(X, n_iter=200)

# 평가: 가장 가까운 평균방향으로 할당
assign = torch.argmax(model.log_prob_components(X), dim=1)
acc = (assign == y).float().mean().item()
print(f"clustering acc ~ {acc:.3f}")
```

## 실전 체크리스트
- 임베딩은 반드시 $$L^2$$-정규화합니다. vMF는 단위구면에서 정의됩니다.[1]
- $$\kappa$$ 추정은 $$A_d(\kappa)$$ 역문제가 핵심입니다. 고정밀도/근사식을 준비하세요.[4][5]
- 큰 $$\kappa$$에서는 $$A_d(\kappa)\approx 1-\frac{d-1}{2\kappa}$$, 작은 $$\kappa$$에서는 테일러 근사로 초기값을 잡으면 수렴이 안정적입니다.[5]
- 혼합 모델은 초기화가 중요합니다. k-means on sphere(코사인 거리)로 $$\mu$$ 초기화를 추천합니다.[5]

## 더 읽기
- 위키: 정의, 정규화 상수, 가우시안과의 관계를 간결히 확인할 수 있습니다.[1]
- MvMF 교재/챕터: MLE, EM 절차와 실전 팁이 잘 정리되어 있습니다.[5]
- PyTorch 구현 노트: 수치안정성, 임의정밀 계산, EM/SGD 레시피를 자세히 다룹니다.[15][4]
- 샘플링 레퍼런스/패키지: R vMF, 블로그 구현을 참고하여 샘플러를 개선할 수 있습니다.[9][10]
- 비등방/딥러닝 확장: nivMF, t-vMF 등 심화 아이디어를 살펴보세요.[12][6]

본 글은 링크된 설명과 연구 레퍼런스를 바탕으로 vMF 분포의 이론과 구현, 그리고 딥러닝 응용까지 빠르게 따라할 수 있도록 구성되었습니다.[1][4][5]

[1](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution)
[2](https://freshrimpsushi.github.io/en/posts/2504/)
[3](https://arxiv.org/html/2504.14164v1)
[4](https://www.emergentmind.com/papers/2102.05340)
[5](https://optml.mit.edu/papers/crc_chapter.pdf)
[6](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860423-supp.pdf)
[7](https://orbit.dtu.dk/files/201566871/paper_1_.pdf)
[8](https://arxiv.org/abs/2106.06375)
[9](https://dlwhittenbury.github.io/ds-2-sampling-and-visualising-the-von-mises-fisher-distribution-in-p-dimensions.html)
[10](https://cran.r-project.org/web/packages/vMF/vMF.pdf)
[11](https://www.jstatsoft.org/v58/i10/)
[12](https://github.com/tk1980/tvMF)
[13](https://arxiv.org/html/2207.08321v3)
[14](https://ieeexplore.ieee.org/document/10422861/)
[15](https://arxiv.org/abs/2102.05340)
[16](https://iskim3068.tistory.com/65)
[17](https://ieeexplore.ieee.org/document/10726517/)
[18](https://www.semanticscholar.org/paper/8015fa4c8335c1c5c7e282eae2a6fbb3de1fc011)
[19](https://ieeexplore.ieee.org/document/10854568/)
[20](https://opg.optica.org/abstract.cfm?URI=josaa-41-7-1287)
[21](https://www.semanticscholar.org/paper/b605bf78e66a8014b4503ddeb32b3c9b50905f30)
[22](https://link.springer.com/10.1007/s11222-024-10419-3)
[23](https://academic.oup.com/mnras/article/522/3/3298/7146849)
[24](https://iopscience.iop.org/article/10.3847/2515-5172/ace4c3)
[25](https://ieeexplore.ieee.org/document/10503141/)
[26](https://arxiv.org/pdf/1502.07104.pdf)
[27](https://arxiv.org/pdf/2106.06375.pdf)
[28](http://benthamopen.com/contents/pdf/TOSPJ/TOSPJ-8-39.pdf)
[29](http://arxiv.org/pdf/2212.14591.pdf)
[30](http://arxiv.org/pdf/2404.07358.pdf)
[31](http://arxiv.org/pdf/2004.06328.pdf)
[32](https://arxiv.org/html/2503.09851)
[33](https://arxiv.org/pdf/0712.4166.pdf)
[34](https://www.aimsciences.org/article/exportPdf?id=2f8505bb-2700-4fbd-b264-e1c7fe6875da)
[35](https://arxiv.org/pdf/2306.03364.pdf)
[36](https://uclalemur.com/blog/von-mises-fisher-distributions-in-machine-learning)
[37](https://www.youtube.com/watch?v=2kIGEEzie1M)
[38](https://dlwhittenbury.github.io/ds-1-sampling-and-visualising-the-von-mises-distribution.html)
[39](https://arxiv.org/html/2404.07358v1)
[40](https://jammalam.faculty.pstat.ucsb.edu/html/Some%20Publications/2015_Directional%20Stats-Intro_WileyStatsRef.pdf)
[41](https://www.sciencedirect.com/science/article/pii/S1877705813011685)


# Reference
출처: https://iskim3068.tistory.com/65 [익플루언서 :: ikfluencer:티스토리]

<details>
<summary>Woods(1994) Simulation of the von Mises Fisher Distribution</summary>

# Simulation of the von Mises Fisher Distribution

## 1. 한 문장 요약

이 논문은 Ulrich(1984)의 von Mises-Fisher 분포 시뮬레이션 알고리즘의 오류를 수정하고 개선된 버전 VM*을 제시하며, 이를 확장하여 R³의 회전에 대한 von Mises-Fisher 행렬 분포와 4차원 이상에서의 Bingham 분포 시뮬레이션 방법을 개발한 연구다.[1]

## 2. 연구의 목적과 필요성

### 연구의 목적
1. **Ulrich 알고리즘의 오류 수정**: 기존에 발표된 Ulrich의 von Mises-Fisher 분포 시뮬레이션 알고리즘 VM이 실제로는 작동하지 않는 문제를 발견하고 이를 수정한 대안 알고리즘 VM*을 제시하는 것[1]

2. **고차원 분포 시뮬레이션 확장**: von Mises-Fisher 행렬 분포(R³ 회전에 해당)와 4차원 단위구에서의 Bingham 분포 시뮬레이션을 위한 새로운 방법론을 개발하는 것[1]

3. **실용적 응용 범위 확대**: 3차원 이상의 방향 통계학 방법론의 실제 적용을 위한 효율적인 시뮬레이션 도구를 제공하는 것[1]

### 연구의 필요성
1. **기존 알고리즘의 실용성 문제**: Ulrich의 원래 알고리즘이 이론적으로는 훌륭했지만 실제 구현에서 작동하지 않는 치명적인 문제가 있어 이를 해결할 필요가 있었음[1]

2. **고차원 방향 통계의 한계**: 3차원 이상에서의 방향 통계학 방법론 응용이 제한적이었으며, 특히 회전 통계 분석과 같은 4차원 효과 응용에 대한 시뮬레이션 방법이 부족했음[1]

3. **수치적 검증의 필요**: 제안된 알고리즘들의 정확성과 효율성을 대규모 시뮬레이션(200만 개 관측치)을 통해 검증할 필요가 있었음[1]

## 3. 연구 주제, 방법, 결과

### 연구 주제
이 연구는 구형 분포(spherical distributions) 시뮬레이션에 관한 연구로, 특히 von Mises-Fisher 분포와 Bingham 분포의 효율적인 난수 생성 방법 개발에 초점을 맞추었다. 주요 관심 분야는 방향 통계학(directional statistics)과 회전 통계 분석이며, 고차원 단위구에서의 분포 시뮬레이션 기법을 다룬다.[1]

### 연구 방법
1. **수정된 VM* 알고리즘 개발**: Ulrich의 원래 알고리즘을 분석하여 오류를 찾아내고, 베타 분포 변환과 rejection sampling을 활용한 개선된 알고리즘을 설계했다[1]

2. **반복적 시뮬레이션 방법(Algorithm B4)**: 4차원 Bingham 분포 시뮬레이션을 위해 반복적 접근법을 개발했으며, 각 단계에서 VM* 알고리즘을 활용하여 점진적으로 목표 분포에 수렴하는 방식을 채택했다[1]

3. **대규모 수치 실험**: 4개의 서로 다른 분포에 대해 각각 200만 개의 관측치를 생성하여 알고리즘의 정확성과 수렴성을 검증했다[1]

### 연구 결과
1. **알고리즘 검증 성공**: 시뮬레이션 기반 추정치가 원래의 충분통계량과 매우 가까운 값을 보여, VM*와 B4 알고리즘 모두 정확한 결과를 생성함을 확인했다[1]

2. **직렬 상관관계 분석**: 시뮬레이션된 단위벡터들의 직렬 상관관계를 SC2 측도로 분석한 결과, a=25회 반복이면 4개 분포 중 3개에서 충분한 성능을 보임을 확인했다[1]

3. **분포별 성능 차이**: Distribution IV가 가장 빠른 수렴을 보인 반면, Distribution I이 가장 높은 직렬 상관관계를 나타내어 분포 매개변수에 따른 성능 차이를 확인했다[1]

## 4. 결론 및 후속 연구

### 연구자들이 제시한 시사점
1. **실용적 시뮬레이션 도구 제공**: 수정된 VM* 알고리즘과 B4 알고리즘이 von Mises-Fisher 분포와 Bingham 분포의 정확한 시뮬레이션을 가능하게 하여, 방향 통계학 연구의 실용적 도구를 제공했다[1]

2. **고차원 확장 가능성**: 개발된 방법론이 4차원 이상의 고차원 Bingham 분포 시뮬레이션으로 확장 가능함을 시사했다[1]

### 연구자들의 후속 연구 계획
연구자들은 직접적인 후속 연구 계획을 명시하지 않았지만, 다음과 같은 방향을 제시했다:

1. **예비 수치 연구의 필요성**: 일반적으로 직렬 상관함수 SC2에 대한 예비 수치 연구를 수행하는 것이 바람직하다고 언급했다[1]

2. **더 정교한 독립성 검정**: 직렬 상관관계가 0이라고 해서 독립성을 의미하지 않으므로, 더 정교한 검정을 고려할 필요가 있다고 제시했다[1]

### 추가 후속 연구 방향 제시

1. **적응적 반복 횟수 결정 방법 개발**: 현재 고정된 반복 횟수(a=25) 대신, 분포 매개변수와 목표 정확도에 따라 최적의 반복 횟수를 자동으로 결정하는 적응적 알고리즘을 개발하여 계산 효율성을 향상시킬 수 있다.

2. **병렬 처리 기반 고성능 시뮬레이션**: GPU나 멀티코어 CPU를 활용한 병렬 처리 버전의 VM*와 B4 알고리즘을 개발하여 대규모 시뮬레이션의 계산 속도를 획기적으로 개선하고, 실시간 응용이 가능한 시스템을 구축할 수 있다.

3. **혼합 von Mises-Fisher 분포 시뮬레이션**: 단일 모드가 아닌 다중 모드를 가진 혼합 von Mises-Fisher 분포의 효율적 시뮬레이션 방법을 개발하여, 복잡한 방향성 데이터 패턴을 모델링할 수 있는 더욱 유연한 도구를 제공할 수 있다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/700907c2-2605-4f95-a079-1bb43ef55eb9/wood1994.pdf)

</details>
