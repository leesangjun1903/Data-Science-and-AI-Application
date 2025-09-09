# Variational-AutoEncoder와 ELBO(Evidence Lower Bound) 가이드

## 핵심 개요
- VAE는 잠재 변수 모델을 신경망으로 학습하는 방법이며, 목적함수는 **ELBO**를 최대화하는 것으로 정리됩니다.[2][1]
- ELBO는 재구성 항과 **KL 발산** 항의 합으로 해석되며, 재구성-정규화의 균형을 조절합니다.[1][2]
- 학습은 재매개변수화 트릭으로 미분 가능하게 만들며, 가우시안 가정에서 KL은 **폐형식**으로 계산합니다.[3][4][5][2]

## 직관: 왜 VAE인가
- 오토인코더는 입력을 잠재공간에 압축하고 복원합니다. 그러나 잠재분포를 명시적으로 학습하지 않으면 표본 생성이 어렵습니다.[1]
- VAE는 잠재공간에 사전분포(보통 표준 정규분포)를 두고, 근사 사후 $q_ϕ(z|x)$ 를 학습하여 샘플링 기반 생성을 가능하게 합니다 [1][2].  
- 결과적으로 임의 $z \sim p(z)$ 에서 디코더로 샘플을 생성할 수 있어 생성 모델로 활용됩니다.[2][1]

## 수학: ELBO 유도 핵심
- 목표는 주변우도 최대화 $$\log p_{\theta}(x) $$ 입니다. 사후 $$p_{\theta}(z|x) $$ 가 난해하므로 변분 근사 $$q_{\phi}(z|x) $$를 도입합니다 [1][2].  
- 표준 전개:  

$$
  \log p_{\theta}(x)=\mathcal{L}(\theta,\phi;x)+D_{\mathrm{KL}}\left(q_{\phi}(z|x)\|p_{\theta}(z|x)\right)
  $$  

$$
  \mathcal{L}(\theta,\phi;x)=\mathbb{E}_{q_{\phi}(z|x)}\left[\log p_{\theta}(x|z)\right]-D_{\mathrm{KL}}\left(q_{\phi}(z|x)\|p(z)\right)
  $$  
  
  여기서 $$\mathcal{L} $$가 **ELBO**이며, 이를 최대화하면 $$\log p_{\theta}(x) $$의 하한을 올리고, 동시에 사후 근사와 실제 사후의 KL을 줄입니다.[4][1]
- 직관: 첫 항은 재구성 가능도를 최대화(픽셀/토큰 수준의 우도), 두 번째 항은 잠재공간을 사전과 정렬시켜 과적합과 병목을 제어합니다.[2][1]

## 재매개변수화 트릭
- 문제: $$\nabla_{\phi}\mathbb{E}\_{q_{\phi}(z|x)}[f(z)] $$는 분포가 $$\phi$$에 의존하여 직접 미분이 어렵습니다 [3][2].  
- 해법: **Reparametrization Trick** : $$z = \mu_{\phi}(x)+\sigma_{\phi}(x)\odot \epsilon,\ \epsilon\sim\mathcal{N}(0,I) $$ 로 무작위성을 입력 측으로 옮겨, “기대의 기울기 = 기울기의 기대” 형태로 몬테카를로 추정이 가능해집니다.[3][2]
- 효과: 그래디언트 분산이 낮고, 자동미분 그래프로 안정적 최적화가 가능합니다. 로컬 재매개변수화 등 변형도 있습니다.[6][2]

## 가우시안 KL의 폐형식
- $$q_{\phi}(z|x)=\mathcal{N}(\mu,\mathrm{diag}(\sigma^2)),\ p(z)=\mathcal{N}(0,I) $$이면  

$$
  D_{\mathrm{KL}}(q\|p)=\frac{1}{2}\sum_{j=1}^{d}\left(\mu_j^2+\sigma_j^2-\log \sigma_j^2-1\right)
  $$  
  
  로 계산되어 역전파가 간단합니다.[5][7][8][4]
- 이 폐형식은 단일 샘플의 몬테카를로 추정으로도 일관된 학습을 돕습니다.[4][2]

## ELBO의 해석과 변형
- ELBO는 정보 이론적으로 엔트로피들의 결합으로 해석될 수 있으며, 재구성/정규화 트레이드오프의 명확한 이해를 제공합니다.[9]
- β-VAE는 KL 항을 $$\beta $$로 가중하여 분해도를 높이지만, 확률론적 엄밀성 관점의 논의도 존재합니다(설계 의도와 실제 효과를 구분해서 적용).[10]
- 실전에서는 우도모형 선택(가우시안/베르누이/디스크리트)과 스케일링이 최적화 난이도에 큰 영향을 줍니다.[1][2]

## PyTorch 구현: 연구 수준 템플릿
- 설계 요점:  
  - Encoder는 $$\mu(x), \log\sigma^2(x) $$를 출력합니다. 수치 안정성을 위해 로그분산을 예측하는 것이 일반적입니다.[2][1]
  - Decoder는 $$p_{\theta}(x|z) $$의 모수(가우시안이면 평균, 베르누이면 로짓)를 출력합니다 [2][1].  
  - 손실은 재구성 음의 로그우도 + KL 폐형식을 사용합니다.[5][4]
  - 재매개변수화로 샘플 효율을 높이고, 단일 샘플 추정이 관행입니다.[3][2]

```python
# 연구용 VAE 구현 템플릿 (MNIST/그레이스케일 가정)
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_dim=784, h_dim=512, z_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)  # log σ^2
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, out_dim=784, h_dim=512, z_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, out_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        logits = self.fc_out(h)  # Bernoulli logits for binary MNIST
        return logits

class VAE(nn.Module):
        def __init__(self, in_dim=784, h_dim=512, z_dim=32):
            super().__init__()
            self.enc = Encoder(in_dim, h_dim, z_dim)
            self.dec = Decoder(in_dim, h_dim, z_dim)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps  # z = μ + σ ⊙ ε

        def forward(self, x):
            mu, logvar = self.enc(x)
            z = self.reparameterize(mu, logvar)
            logits = self.dec(z)
            return logits, mu, logvar

def elbo_loss(logits, x, mu, logvar, beta=1.0, recon_type='bernoulli'):
    x = x.view(x.size(0), -1)
    if recon_type == 'bernoulli':
        # Negative log-likelihood under Bernoulli
        recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='sum')
    elif recon_type == 'gaussian':
        # Assuming unit variance: N(x_hat, I), logits=mean
        recon_loss = 0.5 * F.mse_loss(torch.sigmoid(logits), x, reduction='sum')
    else:
        raise ValueError

    # Closed-form KL(q||p) for diagonal Gaussians
    # 0.5 * sum(μ^2 + σ^2 - log σ^2 - 1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl
    return loss, recon_loss, kl
```
이 구현은 $q_ϕ(z|x)$ 의 가우시안 가정, 재매개변수화, Bernoulli 우도 기반 재구성 손실, 가우시안 KL 폐형식을 모두 반영합니다 [2][3][4][5].  
실험 시 β 조정, z 차원, 은닉 크기, 우도 모형(베르누이/가우시안)의 선택으로 표현력과 분해도, 재구성 품질을 탐색합니다.[10][1][2]

## 학습 루프 예시
```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

model = VAE(z_dim=32).to(device)
opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

for epoch in range(1, 21):
    model.train()
    total, rec_total, kl_total = 0.0, 0.0, 0.0
    for x, _ in train_loader:
        x = x.to(device)
        logits, mu, logvar = model(x)
        loss, rec, kl = elbo_loss(logits, x, mu, logvar, beta=1.0, recon_type='bernoulli')

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()

        total += loss.item()
        rec_total += rec.item()
        kl_total += kl.item()

    print(f"Epoch {epoch}: loss={total/len(train_ds):.4f}, recon={rec_total/len(train_ds):.4f}, kl={kl_total/len(train_ds):.4f}")
```
실전 팁: AdamW, 그래디언트 클리핑, β 스케줄링, 단일 샘플(ε=1) 추정으로 시작하는 것이 안정적입니다. 입력 스케일과 우도 모형의 일치(0-1 스케일↔베르누이)를 확인하세요.[3][1][2]

## 샘플링과 잠재 조작
```python
@torch.no_grad()
def sample(model, n=16, z_dim=32):
    model.eval()
    z = torch.randn(n, z_dim, device=device)  # p(z)=N(0, I)
    logits = model.dec(z)
    x_hat = torch.sigmoid(logits).view(n, 1, 28, 28)
    return x_hat

@torch.no_grad()
def reconstruct(model, x):
    model.eval()
    logits, mu, logvar = model(x.to(device))
    x_rec = torch.sigmoid(logits).view_as(x)
    return x_rec
```
잠재공간 선형 보간, 특정 차원의 조작으로 해석 가능성을 탐색할 수 있습니다. β-VAE 세팅은 분해도를 높여 잠재 의미를 분리하는 데 도움을 줍니다(응용 시 주의).[10][1][2]

## 안정적 학습 체크리스트
- 포스터리어 붕괴 방지: 강한 디코더(CNN/Transformer)에서는 KL annealing(0→1), free-bits(차원별 최소 KL) 기법을 고려합니다.[1][2]
- 로짓-스케일 정합: 베르누이면 입력을 로짓-스케일로 정규화하고 ($[\text{logit}(p) = \log \left( \frac{p}{1-p} \right)]$) BCE-with-logits를 사용합니다. 가우시안이면 분산 파라미터화나 스케일 고정이 필요합니다.[11][2][1]
- z 차원과 β: z를 너무 크게 두면 KL이 과도하게 커져 사전과의 정렬이 어려워질 수 있습니다. β 조절로 표현력과 정규화를 균형화합니다.[10][1]

### Posterior collapse
Posterior collapse는 Variational Autoencoder(VAE)에서 인코더가 생성하는 잠재 변수의 후방 분포 $( q_\phi(z|x) )$ 가 사전 분포 $( p(z) )$ 에 가까워지면서, 잠재 변수 ( z )가 입력 데이터 ( x )에 대한 정보를 담지 못하는 현상입니다. 즉, 디코더가 ( z ) 정보를 무시하고 단순히 사전 분포에서 뽑은 값만으로 데이터를 재구성하는 상황입니다.

수식으로 설명하면, VAE의 목적 함수는 다음과 같습니다:

```math
[\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x) || p(z))
]
```

여기서 $(\mathrm{KL}(\cdot || \cdot))$ 는 쿨백-라이블러 발산입니다. 만약 디코더가 너무 강력해서 ( z ) 없이도 데이터를 잘 재구성할 수 있다면, 인코더가 $( q_\phi(z|x) )$ 를 $( p(z) )$ 와 같게 만들고 ( z )에서 얻는 정보가 없어집니다. 따라서 $(\mathrm{KL}(q_\phi(z|x) || p(z)))$ 가 0에 가까워지고, posterior 분포가 prior로 ‘collapse’됩니다.

이 현상의 주요 원인은:

- 너무 약한 인코더 분포: 잠재 변수의 분포 $( q_\phi(z|x) )$ 가 입력 ( x )와 상관 없이 거의 일정한 분포가 되어 정보 전달이 안 됨 (too weak).

- 너무 표현력 강한 디코더: autoregressive 디코더가 latent 변수 없이 데이터를 복원할 가능성이 높아져 디코더가 ( z )를 무시함.

- 불안정하거나 노이즈가 심한 인코더 분포: $( q_\phi(z|x) )$ 가 불안정하거나 일정치 않음 (too noisy).

이로 인해 모델이 latent 변수를 무시하고 오토인코더처럼 작동하게 되어, 잠재 공간의 의미가 사라지게 됩니니다.

이를 방지하기 위해 KL 가중치를 점차 높이는 KL annealing이나 vector quantisation(VQ-VAE) 기법 등이 사용됩니다.

#### KL Annealing
KL Annealing은 Variational Autoencoder(VAE) 학습 시 KL 손실 항에 대한 가중치 β를 점진적으로 증가시키면서 posterior collapse를 방지하는 기법입니다.

수식으로 나타내면 VAE의 손실 함수는 다음과 같습니다:

```math
[\mathcal{L} = \text{Likelihood} + \beta \times \text{KL divergence}
]
```

여기서 β는 학습 초기에는 작게 설정해 잠재 변수(z)가 의미 있는 정보를 학습하도록 하고, 점차 β를 1(또는 최종값)까지 증가시켜 latent 분포를 prior에 맞추도록 합니다.

대표적인 β 증가 식은 다음과 같습니다:

```math
[\beta_{\text{epoch}} = \text{end} + (\text{start} - \text{end}) \times \text{rate}^{\text{epoch}}
]
```

여기서 start는 초기 β 값(보통 0), end는 최종 β 값(보통 1), rate는 증가율입니다.

이 방법은 latent 변수의 표현을 안정적으로 유도해 VAE의 성능을 개선하는 데 효과적이며, 학습 초기에 KL 손실이 너무 커서 latent 변수가 무시되는 문제를 완화합니다.

#### free-bits
free-bits 기법에서 차원별 최소 KL(쿨백-라이블러 발산)는 각 잠재 변수 차원이 가지는 KL 발산의 최소값을 제한하여, 정보 소실 없이 잠재 공간을 효과적으로 활용하도록 하는 방법입니다.

수식적으로, 일반적인 변분 오토인코더(VAE)의 KL 항은 다음과 같습니다:

```math
[D_{KL}(q(z|x) \parallel p(z)) = \sum_{d=1}^D D_{KL}(q(z_d|x) \parallel p(z_d))
]
```

여기서 (D)는 잠재 변수의 차원 개수, $(z_d)$는 (d)번째 차원을 의미합니다.

free-bits 기법은 각 차원당 KL 발산 값에 대해 최소 임계치 $(\lambda)$ 를 설정하여 다음과 같이 적용합니다:

```math
[KL^{free}d = \max \left( D{KL}(q(z_d|x) \parallel p(z_d)), \lambda \right)
]
```

즉, 각 차원의 KL 발산 값이 $(\lambda)$ 보다 작으면 $(\lambda)$ 로 보정하여, 모델이 일부 차원을 충분히 사용하도록 강제합니다. 전체 KL 항은 차원별 보정된 값을 합산한 후 사용됩니다.

이 방법은 특히 잠재 공간의 일부 차원만 활성화되어 나머지 차원이 죽는 현상을 방지하며, 정보 표현의 부족 문제를 완화하여 안정적인 학습을 돕습니다.

#### VQ-VAE(Vector Quantized Variational Autoencoder) : 
VQ-VAE(Vector Quantized Variational Autoencoder)는 인코더가 연속적 잠재 벡터 $( z_e(x) )$ 를 생성하면, 이를 코드북 $( e = {e_j} )$ 에서 가장 가까운 벡터 $( e_k )$ 로 양자화하는 모델입니다. 이 최종 이산 잠재 변수는

```math
[z_q(x) = e_k, \quad \text{where } k = \arg\min_j | z_e(x) - e_j |^2
]
```

로 정의됩니다.

학습 손실 함수는 다음과 같이 세 부분으로 구성됩니다:

```math
[L = \underbrace{\log p(x|z_q(x))}_{\text{재구성 손실}} + \underbrace{| \mathrm{sg}[z_e(x)] - e_k |2^2}_{\text{벡터 양자화 손실}} + \underbrace{\beta | z_e(x) - \mathrm{sg}[e_k] |2^2}_{\text{커밋먼트 손실}}
]
```

여기서 $(\mathrm{sg}[\cdot])$ 는 그래디언트가 흐르지 않도록 하는 stop-gradient 연산자이며, $(\beta)$ 는 하이퍼파라미터입니다. 재구성 손실은 디코더가 원본을 잘 복원하도록 하고, 벡터 양자화 손실과 커밋먼트 손실은 인코더와 코드북 벡터가 서로 잘 맞도록 조정합니다.

이때 인코더 출력의 벡터를 가장 가까운 코드북 벡터로 대체하는 과정이 바로 벡터 양자화이며, 미분 불가능하므로 "straight-through estimator"를 이용해 학습합니다.

요약하면, VQ-VAE는 연속적인 잠재 공간을 이산적인 코드북 벡터로 양자화하여 표현하며, 이산적 표현 학습과 재구성 성능을 동시에 추구하는 생성 모델입니다.

##### Straight-through estimator(STE)
Straight-through estimator(STE)는 신경망에서 비미분가능한 함수(예: 이진화 함수)의 역전파를 가능하게 하는 기법으로, 순전파 때는 실제 비미분 함수 값을 사용하고 역전파 때는 단순히 입력에 대한 그래디언트를 그대로 전달하는 방법입니다.

수식으로 표현하면 다음과 같습니다.

순전파: $( y = \text{sgn}(x) )$ 또는 $( y = \mathrm{round}(x) )$

역전파: $( \frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial y} )$

즉, 역전파에서는 ( y )의 미분값 대신 입력 ( x )의 미분값을 그대로 사용하는 방식입니다. 일부 변형에서는 그래디언트 크기에 제한(saturation effect)을 두기도 합니다.

이 방법은 이진 신경망, 이산화 연산 등에서 효과적으로 사용되며, 실제 미분이 불가능한 함수에서 간단하게 근사해 학습이 가능하도록 합니다. 최근 연구에서는 STE가 1차 근사임을 증명하고, 2차 정확도를 갖는 개선 방법도 제안되고 있습니다.

##### Straight-Through Gumbel-Softmax Estimator
**Straight-Through Estimator (STE)**는 비미분 가능 함수나 이산(discrete) 연산에 대해 신경망의 역전파 시 미분값을 근사하여 전달하는 기법입니다.  
실제 순전파 때는 함수 $( z = \arg\max(x) )$ 같은 비미분 함수를 사용하고, 역전파 때는 미분 가능한 근사값 ( y )의 기울기를 대신 쓰는 방식입니다.

대표적인 수식은 다음과 같습니다:

```math
[\text{forward: } z = \text{one-hot}(\arg\max(x)) \quad , \quad \text{backward: } \frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial y}
]
```

여기서 ( y )는 Gumbel-Softmax 분포와 같이 미분 가능한 연속 근사값입니다. 즉, 역전파 시 ( z )에 대한 기울기를 직접 계산하지 않고, 연속적인 근사 ( y )에 대해 미분값을 전달합니다.

Gumbel-Softmax는 이산 확률 분포를 미분 가능하게 근사하는 대표적 방법이며, STE와 함께 쓰여서 다음과 같이 동작합니다:

###### Gumbel-Softmax
Gumbel-Softmax는 categorical 분포에서 샘플링을 미분 가능하게 하기 위해 argmax를 softmax 함수로 근사하여 사용합니다. 샘플 (y)는 다음과 같이 구합니다:

```math
[y_i = \frac{\exp\left((\log(\pi_i) + g_i)/\tau\right)}{\sum_{j=1}^k \exp\left((\log(\pi_j) + g_j)/\tau\right)}, \quad i=1,\ldots,k
]
```

여기서

- $(\pi_i)$는 범주 i의 확률,
- $(g_i)$는 Gumbel 분포에서 샘플링한 노이즈,
- $(\tau)$는 온도 파라미터(temperature)로 $(\tau \to 0)$ 일 때 argmax 근사,

###### 왜 argmax에 가까워지나?
$(\tau \to 0)$ 일 때, 분자는 가장 큰 $((\log \pi_i + g_i))$ 값에 대해 지수 함수가 크게 증폭되고, 나머지는 0에 가까워지므로 softmax 출력이 거의 원-핫 벡터(one-hot)에 수렴합니다.  
(softmax 함수의 출력은 점점 확률 분포에서 가장 큰 값(최대 로그잇)에 해당하는 위치로 치우쳐 one-hot 벡터(즉, argmax에 가까운 선택)를 하게 됩니다.  
이는 softmax 수식에서 τ가 분모에 있으므로 작은 τ는 분포를 매우 뾰족하게 만들어 가장 큰 값이 지배적이 되기 때문입니다.)

따라서 샘플링이 argmax와 유사하게 되고 미분 불가능한 이산 선택을 연속 근사로 표현해줍니다.  
(argmax 함수는 수학과 프로그래밍에서 함수나 배열에서 **최댓값을 가지는 입력(인덱스)**을 찾는 함수입니다. 즉, 함수 $( f(x) )$ 에서 $( f(x) )$ 가 최대가 되는 ( x ) 값을 반환합니다.)

이 원리는 VQ-VAE 같은 discrete latent 변수 모델에서 부드러운 샘플링에서 점점 결정론적인 선택으로 전환할 때 활용됩니다.  
온도 감소(annealing)를 통해 모델 학습 시 연속적인 확률분포에서 점차 argmax처럼 확실한 one-hot 선택(이 벡터는 하나의 원소만 1이고 나머지는 모두 0인 (n)차원의 벡터로, 데이터의 각 고유 범주를 서로 구분할 수 있게 만듭니다.)으로 수렴시키는 것입니다.

따라서, $(\tau \to 0)$ 로 갈수록 softmax는 argmax에 가까워져 one-hot 벡터에 근접합니다. 이 수식 덕분에 sampling 동작에 역전파가 가능해집니다.

- 순전파에서 $( z = \arg\max \approx (\text{softmax}((\log \pi + g)/\tau)) )$ 를 사용해 원-핫 벡터 선택
- 역전파에서는 ( z ) 대신 부드러운 확률 분포 $( y = \text{softmax}((\log \pi + g)/\tau) )$ 의 기울기 사용

여기서 ( g )는 Gumbel 잡음, $(\tau)$ 는 온도 파라미터로 작을수록 이산적 분포에 가까워집니다.

즉, ST Gumbel-Softmax Estimator는

```math
[\text{forward: } z = \arg\max \quad,\quad \text{backward: } \frac{\partial L}{\partial z} := \frac{\partial L}{\partial y}
]
```

로 정의되며 $(\tau)$ 조절에 따라 편향된 미분 근사가 발생할 수 있어 적절한 값 선택이 중요합니다.

요약하면, STE는 이산 출력을 가진 모델을 학습할 때 순전파는 이산 연산을 하고 역전파는 미분 가능한 연속 근사로 진행하는 기법이며, Gumbel-Softmax는 이와 결합해 이산 확률 선택의 부드러운 미분 근사를 가능하게 합니다.

> Gumbel distribution

Gumbel distribution는 주로 여러 샘플 중 최댓값 또는 최솟값의 분포를 모델링하는 연속 확률분포로, 위치 파라미터 $(\mu)$와 척도 파라미터 $(\beta)$ 두 가지로 정의됩니다. 자연재해의 극단적 사건(예: 홍수, 지진) 분석에 주로 활용됩니다.

확률밀도함수(PDF)는

```math
[f(x;\mu,\beta) = \frac{1}{\beta} \exp\left(-\frac{x-\mu}{\beta} - e^{-\frac{x-\mu}{\beta}}\right)
]
```

이고, 누적분포함수(CDF)는
```math
[F(x;\mu,\beta) = \exp\left(-e^{-\frac{x-\mu}{\beta}}\right)
]
```
입니다.

표준 Gumbel distribution 은 $(\mu=0), (\beta=1)$ 이며, 평균은 오일러-마스케로니 상수 $(\gamma \approx 0.5772)$, 분산은 $(\frac{\pi^2}{6})$ 입니다. 이는 종종 극단값 이론에서 "극값 분포 유형 I"으로도 불립니다.

요약하면, Gumbel distribution 은 극단적인 이벤트의 확률을 예측하는 데 유용한 분포로, 최대값(또는 최소값)의 분포를 통계적으로 잘 나타냅니다.

## 더 깊게 보기: 이론 노트
- ELBO는 사전의 음의 엔트로피, 관측 우도의 기대 음의 엔트로피, 변분 분포의 평균 엔트로피 합으로 해석되는 수렴 형태를 가집니다. 이는 모델링 가정의 결과적 한계를 이해하는 데 유용합니다.[9]
- 잠재/우도 분포의 대안(예: heavy-tailed, Student-t)은 경계 사례 데이터에 유리할 수 있으며 최근 변형이 제안됩니다.[12]
- 변분/재매개변수화의 기본 레퍼런스와 최신 튜토리얼은 개념적 맥락과 실전 팁을 균형 있게 제공합니다.[13][2][3][1]

### ELBO 수렴 형태 :
ELBO (Evidence Lower Bound)는 다음과 같이 수식으로 표현되며, 사전(prior)의 음의 엔트로피, 관측 우도의 기대 음의 엔트로피, 변분 분포(variational distribution)의 평균 엔트로피로 해석할 수 있습니다.

```math
[\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D{\mathrm{KL}}(q(z|x) | p(z))
]
```

여기서,

$(\mathbb{E}_{q(z|x)}[\log p(x|z)])$ 는 관측 우도(log-likelihood)의 기대값으로, 복원 오류(reconstruction error)의 음의 엔트로피에 해당합니다.

$(D_{\mathrm{KL}}(q(z|x) | p(z)))$ 는 변분 분포 $(q(z|x))$ 와 사전 분포 $(p(z))$ 사이의 **쿨백-라이블러 발산(KL divergence)**이며, KL 발산은 다음과 같이 엔트로피 차이로 분해됩니다:

```math
[D_{\mathrm{KL}}(q(z|x) | p(z)) = -H(q(z|x)) - \mathbb{E}_{q(z|x)}[\log p(z)]
]
```

따라서 ELBO는 다음과 같이 쓰여 엔트로피 관점에서 해석할 수 있습니다.

```math
[\text{ELBO} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{관측 우도의 기대 음의 엔트로피}} + \underbrace{H(q(z|x))}_{\text{변분 분포의 평균 엔트로피}} + \underbrace{\mathbb{E}_{q(z|x)}[\log p(z)]}_{\text{사전의 음의 엔트로피}}
]
```

즉, ELBO는 사전 확률 $(p(z))$의 로그 확률(음의 엔트로피), 관측 데이터에 대한 모델 우도 $(\log p(x|z))$의 기대값, 그리고 변분 분포 $(q(z|x))$의 엔트로피 합으로 해석되며, 이 값을 최대화하는 것이 변분 추론의 목표입니다.

### Heavy-tailed Distribution
Heavy-tailed 분포는 꼬리가 지수함수보다 느리게 감쇠되어 극단적인 값들이 자주 발생하는 분포를 의미합니다. 수식적으로는 분포 함수 (F(x))의 우측 꼬리가 임의의 $(\mu > 0)$에 대해 다음과 같이 지수감쇠 보다 느리게 감소할 때로 정의됩니다:

```math
[\limsup_{x \to \infty} \frac{1 - F(x)}{e^{-\mu x}} = \infty
]
```

즉, 확률 $(\Pr(X > x))$가 지수감쇠보다 천천히 감소하는 경우입니다.

특히, 멱법칙(power-law) 형태의 heavy-tailed 분포는 확률밀도함수(pdf)가 다음과 같이 표현됩니다:

```math
[f_X(x) = C x^{\alpha}, \quad x \in [x_0, x_1]
]
```

여기서 (C)는 적분이 1이 되도록 하는 정규화 상수이고, $(\alpha < -1)$인 경우 꼬리가 두꺼워집니다. 이런 분포는 꼬리가 매우 길고, 극단적인 사건이 나타날 확률이 높습니다.

대표적인 heavy-tailed 분포로는 파레토 분포, 코시 분포, 로그정규분포, 안정분포 등이 있습니다.

### Student's t Distribution
Student's t 분포의 확률 밀도 함수(PDF)는 자유도 $( \nu )$ 를 가지며 다음과 같이 정의됩니다:

```math
[f(t) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\sqrt{\pi \nu} \Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu + 1}{2}}
]
```

여기서 $( \Gamma(\cdot) )$ 는 감마 함수이고, $( \nu )$ 는 자유도입니다. 이 함수는 표준정규분포와 비슷하지만, 자유도가 작을 때 꼬리가 더 두꺼운 분포를 나타냅니다.


## 참고 문헌
- Doersch, Tutorial on VAEs: 직관과 수식, 실무 포인트를 균형 있게 정리한 고전 튜토리얼입니다.[1]
- Kingma & Welling, Auto-Encoding Variational Bayes: 원 논문과 저자의 확장 노트는 구현과 이론의 표준 참조입니다.[14][2]
- 재매개변수화 트릭의 공식적 해설과 KL 폐형식 도출 자료는 구현의 정확성을 높여줍니다.[8][4][5][3]

부록: 핵심 수식 모음
- ELBO 분해:  

$$
  \mathcal{L}(x)=\mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)]-D_{\mathrm{KL}}(q_{\phi}(z|x)\|p(z)) \quad \text{with}\ \log p_{\theta}(x)\ge \mathcal{L}(x)
  $$

[4][1]
- 재매개변수화:  

$$
  z=\mu_{\phi}(x)+\sigma_{\phi}(x)\odot \epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
  $$

[2][3]
- 가우시안 KL(대각):  

$$
  D_{\mathrm{KL}}=\frac{1}{2}\sum_{j}\left(\mu_j^2+\sigma_j^2-\log \sigma_j^2-1\right)
  $$

[8][5][4]

추천 읽을거리
- Variational Inference 배경과 ELBO 맥락 정리(국문): 변분추론-ELBO 연결을 MLE/MAP 관점에서 설명합니다.[15][16][17]
- VAE 훈련 실전 노하우: KL 해석, 트레이드오프, 하이퍼 감각을 다룹니다.[18][2][1]
- ELBO 성질과 변형: 수렴 성질, β-가중의 이론적 논의, 최근 확장들을 참고하세요.[12][9][10]

[1](https://arxiv.org/pdf/1606.05908.pdf)
[2](https://arxiv.org/pdf/1906.02691.pdf)
[3](https://gregorygundersen.com/blog/2018/04/29/reparameterization/)
[4](https://arxiv.org/pdf/1907.08956.pdf)
[5](https://datahacker.rs/gans-004-variational-autoencoders-in-depth-explained/)
[6](https://arxiv.org/abs/1506.02557)
[7](https://neurips.cc/media/neurips-2023/Slides/70438_2ls79ol.pdf)
[8](https://johfischer.com/2022/05/21/closed-form-solution-of-kullback-leibler-divergence-between-two-gaussians/)
[9](https://arxiv.org/pdf/2010.14860.pdf)
[10](https://arxiv.org/abs/2312.06828)
[11](https://hugrypiggykim.com/2018/09/07/variational-autoencoder%EC%99%80-elboevidence-lower-bound/)
[12](http://arxiv.org/pdf/2312.01133.pdf)
[13](https://arxiv.org/pdf/2208.07818.pdf)
[14](https://arxiv.org/abs/1312.6114)
[15](https://process-mining.tistory.com/161)
[16](https://modulabs.co.kr/blog/variational-inference-intro)
[17](https://kh-mo.github.io/notation/2019/09/30/elbo/)
[18](https://arxiv.org/abs/2309.13160)
[19](http://arxiv.org/pdf/1912.10309.pdf)
[20](https://arxiv.org/pdf/2005.10686.pdf)
[21](https://arxiv.org/html/2408.16883)
[22](http://arxiv.org/pdf/1910.13181.pdf)
[23](https://arxiv.org/html/2410.07840)
[24](http://arxiv.org/pdf/2407.06797.pdf)
[25](https://arxiv.org/pdf/2106.15921.pdf)
[26](https://arxiv.org/pdf/2408.08931.pdf)
[27](https://arxiv.org/pdf/2210.15407.pdf)
[28](http://arxiv.org/pdf/2002.03147.pdf)
[29](http://arxiv.org/pdf/2105.10867.pdf)
[30](https://arxiv.org/pdf/2212.04451.pdf)
[31](https://arxiv.org/pdf/1911.02469.pdf)
[32](https://ai-com.tistory.com/entry/DL-VAE-Variational-Autoencoder-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)
[33](https://glanceyes.com/entry/VAE-%EA%B3%84%EC%97%B4-%EB%AA%A8%EB%8D%B8%EC%9D%98-ELBOEvidence-Lower-Bound-%EB%B6%84%EC%84%9D)
[34](https://velog.io/@jojo0217/VAE-ELBO-%EC%84%A4%EB%AA%85)
[35](https://simpling.tistory.com/34)
[36](https://bigdata-analyst.tistory.com/263)
[37](https://jmtomczak.github.io/blog/4/4_VAE.html)
[38](https://leenashekhar.github.io/2019-01-30-KL-Divergence/)
[39](https://gregorygundersen.com/blog/2021/04/16/variational-inference/)

https://hugrypiggykim.com/2018/09/07/variational-autoencoder%EC%99%80-elboevidence-lower-bound/
