# Lottery Ticket Hypothesis 가이드

핵심 요약: Lottery Ticket Hypothesis(LTH)는 큰 신경망 안에 이미 성능 좋은, 매우 **희소한 서브네트워크**(winning ticket)가 존재하며, 이를 찾아 단독으로 학습해도 원래 모델과 비슷하거나 더 좋은 성능을 낼 수 있다는 가설입니다. 이 글은 LTH의 개념과 맥락, 실무적 장단점, 대표 후속 연구 흐름, 그리고 대학생도 바로 시도해볼 수 있는 PyTorch 예시 코드까지 제공합니다.[1][2]

## 왜 LTH가 중요한가
- 거대 모델은 학습·추론 비용이 큽니다. LTH는 학습 가능한 희소 서브네트워크를 찾아 파라미터·연산량을 줄이면서도 성능 저하 없이 학습할 가능성을 제시합니다.[2][1]
- 핵심 통찰은 “랜덤 초기화된 밀집 네트워크 내부에, 운 좋게 초기화가 맞아 학습이 잘 되는 연결만 모은 서브네트워크가 있다”는 점이며, 이를 winning ticket이라 부릅니다.[3][2]

## 기본 개념 정리
- Winning ticket: 초기화가 “당첨”되어, 같은 학습 예산으로 원본과 유사/우수 성능에 도달하는 희소 서브네트워크입니다.[4][2]
- Iterative Magnitude Pruning(IMP): 학습→가중치 크기(magnitude) 기준 프루닝→초기화로 되돌림(또는 초기 단계로 리와인드)→재학습을 반복해 mask를 정교화하는 절차입니다.[5][6]
- One-shot vs Iterative: 한 번에 큰 비율을 자르는 one-shot보다, 여러 번 작게 자르는 iterative가 성능상 유리합니다(레이어 붕괴 방지 등).[6][5]

## 대표 알고리즘 흐름
- IMP 알고리즘 요약: (1) 랜덤 초기화 W0, (2) 전체 학습 T, (3) 작은 |W|부터 p% 제거, (4) 남은 연결만 초기값(W0 또는 초반 k 스텝 리와인드 Wk)으로 되돌림, (5) 재학습, (6) 반복합니다[5][2].  
- 중요한 변형: “리와인드(rewinding)”는 완전 초기화 대신 학습 초반 k 스텝의 파라미터 또는 학습률 상태로 되돌려 안정성을 높입니다.[7][5]
- 리와인딩은 일반적으로 딥러닝 모델에서 *가중치(weight)*나 *상태(state)*를 다시 되돌리거나 재설정하는 과정을 의미합니다.

### Iterative Magnitude Pruning(IMP)
Iterative Magnitude Pruning(IMP)은 신경망의 가중치 중에서 절대값이 작은 것들을 반복적으로 제거(프루닝)하고, 다시 학습하며 성능을 유지하거나 향상시키는 알고리즘입니다. 일반적으로 다음 세 단계로 진행됩니다: 1) 전체 데이터로 모델을 학습, 2) 가중치 절대값 기준으로 중요하지 않은 가중치를 제거, 3) 남은 가중치로 모델을 다시 미세 조정하는 과정을 여러 번 반복합니다.

IMP는 최초 학습 후 가중치를 기준 임계값에 따라 마스킹하여 희소화하고, 이후 초기값 또는 중간값으로 "리와인딩"하여 다시 학습하는 과정을 여러 반복(iteration) 수행해 sparsity(희소성)를 높여나갑니다. 이렇게 반복 훈련과 프루닝, 리와인딩을 거치면서 안정적인 희소 subnetworks (소위 ‘winning tickets’)를 찾아냅니다.

Iterative Magnitude Pruning(IMP)에서 **mask(마스크)**는 신경망의 각 파라미터별로 해당 파라미터를 유지할지(1) 버릴지(0) 결정하는 이진 벡터입니다.  
즉, 파라미터 크기가 특정 임계값(τ) 이상이면 마스크 값이 1, 그렇지 않으면 0으로 설정하여 희소한(subnetwork) 모델을 만듭니다.

마스크의 역할은 주어진 네트워크에서 어떤 파라미터를 남기고 어느 파라미터를 제거할지 지정하는 ‘필터’ 역할을 하며, 결국 성능을 유지하는 희소 신경망(“winning ticket”)을 찾도록 돕습니다.

기존의 한 번에 큰 폭으로 프루닝하는 원샷 프루닝보다 IMP는 여러 번 반복하면서 손실을 최소화하고, 층별 파라미터 손실 문제(layer-collapse)를 방지하는 특징이 있습니다. 다만 반복 학습을 여러 번 하므로 계산 비용이 비교적 높다는 단점도 존재합니다.

요약하면, IMP는 가중치 크기에 따른 반복적 제거와 재학습, 초기값 리와인딩을 통해 효과적으로 희소 신경망을 탐색하는 방법입니다.

## 장점과 한계
- 장점: 높은 희소성에서도 원본과 유사 성능, 일부 설정에서 학습 수렴 가속, 추론 효율화 가능성이 보고되었습니다.[1][2]
- 한계: 승리 티켓 탐색 비용이 큽니다(대부분 전체 모델을 한 번은 학습해야 함). 대규모 데이터·모델에서 효율성·확장성 이슈가 남아 있습니다.[3][5]

## 중요한 후속 연구 흐름
- 사전 학습 없이/초기 단계에서의 프루닝  
  - SNIP: 학습 전 연결 민감도(gradient 기반 saliency)로 한 번에 프루닝하여 학습 비용을 크게 절감합니다(단, 상호작용 무시는 한계).[8][9]

### SNIP (Single-shot Network Pruning based on Connection Sensitivity)
**SNIP (Single-shot Network Pruning based on Connection Sensitivity)**는 신경망의 성능 저하 없이 불필요한 연결(weight)을 한 번에 제거하는 방법입니다. 학습 전에 네트워크의 각 연결이 손실(loss)에 미치는 영향을 미분(derivative)하여 중요도를 측정하고, 중요도가 낮은 연결을 제거하는 방식입니다.

기존 반복적이고 복잡한 pruning 과정과는 달리 SNIP는 초기화된 네트워크에서 단 한번만 연결 감도를 계산하여 구조적으로 중요한 연결만 남기고 나머지를 제거합니다. 이렇게 하면 사전 학습이나 복잡한 스케줄 없이도 성능이 거의 유지되는 희소(sparse) 네트워크를 얻을 수 있습니다. 이 방법은 합성곱, 잔차, 순환 신경망 등 다양한 아키텍처에도 적용 가능합니다.

#### Gradient 기반 saliency : Connection Sensitivity
SNIP (Single-shot Network Pruning) 방법에서 gradient 기반 saliency는 각 연결의 중요도를 측정하기 위해 손실 함수에 대한 가중치의 미분값, 즉 connection sensitivity를 활용합니다. 이는 네트워크가 학습되기 전, 초기화된 상태에서 각 가중치를 제거할 때 손실이 얼마나 변하는지를 계산하여 중요하지 않은 연결을 한 번에 제거하는 방식입니다.

구체적으로, 연결의 중요도는 다음과 같이 정의됩니다:

네트워크의 손실 함수 ( $\mathcal{L}$ )에 대해 가중치 ( $w_i$ )를 0으로 만들었을 때 손실이 증가하는 정도를 미분값 ( $\left| \frac{\partial \mathcal{L}}{\partial w_i} \cdot w_i \right|$ ) 로 측정합니다.
이 값이 작을수록 해당 연결이 네트워크 성능에 덜 영향을 준다고 판단하여 pruning 대상이 됩니다.

이 방식의 장점은:

- 훈련 전 단 한번만 계산하여 pruning 하므로 복잡한 반복적 pruning이나 추가 하이퍼파라미터 조절이 필요 없습니다.
- 초기 가중치와 입력 배치에서 손실의 기울기를 활용해 데이터 의존적인 중요도를 측정해 네트워크 구조에 강건합니다.
- Pruning 후, 나머지 중요 연결들만 남긴 희소 네트워크를 일반적인 방식으로 학습시킵니다.

즉, SNIP은 gradient(기울기)를 이용해 각 연결의 민감도(impact)를 평가하는 connection sensitivity라는 saliency 기준으로, 학습 전에 한 번에 중요하지 않은 연결을 제거하는 단일 단계의 네트워크 pruning 기법입니다.

요약하면, SNIP는:

- 학습 전에 진행하는 single-shot pruning 기법
- 연결 하나하나의 손실에 대한 민감도(연결 감도)를 자동 미분으로 계산
- 중요 연결만 보존하고 나머지 제거
- 간단하고 빠르며 기존 방법 대비 확장성이 좋음
이로 인해 높은 희소률에서도 원래 네트워크와 비슷한 정확도 유지가 가능합니다.

- 초기 단계에서 티켓 조기 탐지  
  - Early-Bird/Mask 안정화 관찰: 학습 초반에 마스크가 수렴하므로 조기 중단으로 비용 절감 가능성이 보고됩니다(개념 리뷰 맥락).[1][3]
- 리와인드와 파인튜닝 비교  
  - 프루닝 후 파인튜닝 대신 가중치/학습률 리와인드가 동등하거나 유리한 경우가 보고되었습니다.[7][5]
- 이론·서베이  
  - 최근 서베이는 LTH 검증과 함께 효율성·스케일링 과제를 정리하며 표준화 필요성을 제기합니다.[10][1]

## 실전 적용 팁
- 데이터·모델 크기가 크면 IMP 반복 수를 제한하고, 프루닝 비율을 완만하게 시작하세요(예: 10–20% 포인트씩).[5][7]
- 리와인드: 완전 초기화 대신 초반 k 스텝 파라미터로 되돌리면, 학습 안정성과 재현성이 좋아집니다.[7][5]
- 초기 신호 활용: SNIP 같은 “훈련 전” 기준으로 1차 마스크를 잡고, 소규모 파인튜닝으로 보정하는 하이브리드 전략도 실용적입니다.[11][8]

## PyTorch 예시 코드: CIFAR-10, IMP로 Winning Ticket 찾기
아래는 교육용 기준으로 단순화한 레퍼런스입니다. 핵심은 “마스크를 유지하며 학습”, “프루닝은 magnitude 기준”, “리와인드 시점(k) 선택”입니다.[2][5]

### Magnitude Pruning
신경망에서 가중치의 절대값(magnitude)을 기준으로 중요도가 낮은 가중치들을 제거하는 방법입니다. 절대값이 작은 가중치들을 선택하여 0으로 만들어 네트워크를 희소(sparse)하게 하여 모델 크기를 줄이고 계산량을 감소시키는 방식입니다.

```python
# PyTorch 2.x
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy

# 1) 간단한 CNN
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(8*8*64, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))       # 32x32
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 16x16 -> 16x16 (stride2)
        x = F.max_pool2d(x, 2)          # 8x8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 2) 데이터
tfm = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

# 3) 유틸: 평가
def evaluate(net, loader, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total

# 4) 마스크 초기화/적용
def get_masks(model):
    masks = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() > 1:  # 가중치만 (bias 제외 권장)
            masks[name] = torch.ones_like(p, dtype=torch.bool)
    return masks

def apply_masks(model, masks):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                p.mul_(masks[name].float())

def masked_parameters(model, masks):
    for name, p in model.named_parameters():
        if name in masks:
            yield p

# 5) magnitude 기반 프루닝
def magnitude_prune(model, masks, prune_ratio):
    # 모든 남은 가중치의 절대값을 모아 전역 임계값 계산 (global pruning)
    all_scores = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                score = p.abs()[masks[name]]
                all_scores.append(score.view(-1))
    if not all_scores:
        return masks
    all_scores = torch.cat(all_scores)
    k = int(prune_ratio * all_scores.numel())
    if k <= 0: 
        return masks
    threshold = torch.kthvalue(all_scores, k).values

    # 임계값 이하를 0으로 마스킹
    new_masks = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                keep = (p.abs() > threshold) & masks[name]
                new_masks[name] = keep
    return new_masks

# 6) 학습 루프(마스크 유지), 리와인드 지원
def train_with_masks(model, masks, optimizer, scheduler, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            # 마스크된 가중치만 업데이트
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in masks:
                        p.grad *= masks[name].float()
            optimizer.step()
            # 마스크 적용으로 0 유지
            apply_masks(model, masks)
        if scheduler:
            scheduler.step()

# 7) IMP 드라이버
def iterative_magnitude_pruning(
    base_model_fn,
    device="cuda",
    total_rounds=5,
    prune_per_round=0.2,
    pretrain_epochs=20,
    retrain_epochs=20,
    rewind_k_epochs=5,  # 리와인드 시점
    lr=0.1
):
    # 초기화
    dense = base_model_fn().to(device)
    dense_init = copy.deepcopy(dense.state_dict())  # W0 저장
    masks = get_masks(dense)

    # 1) 프리트레인 T 에폭
    optimizer = torch.optim.SGD(dense.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)
    train_with_masks(dense, masks, optimizer, scheduler, device, epochs=pretrain_epochs)
    base_acc = evaluate(dense, test_loader, device)

    print(f"Dense pretrain accuracy: {base_acc:.4f}")

    # 리와인드 지점 Wk 저장(초기 몇 에폭만 학습한 상태를 시뮬레이션하려면 별도 러닝 필요)
    # 간단화를 위해 여기서는 W0와 동일하게 두되, 실전에서는 pretrain 중간 체크포인트 사용 권장
    rewind_state = copy.deepcopy(dense_init)

    # 2) IMP 반복
    sparse_model = None
    current_masks = masks
    for r in range(total_rounds):
        # (a) 프루닝
        current_masks = magnitude_prune(dense, current_masks, prune_per_round)

        # (b) 리와인드: 가중치를 Wk로 되돌리고 마스크 적용
        sparse_model = base_model_fn().to(device)
        sparse_model.load_state_dict(rewind_state, strict=True)
        apply_masks(sparse_model, current_masks)

        # (c) 재학습
        optimizer = torch.optim.SGD(sparse_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=retrain_epochs)
        train_with_masks(sparse_model, current_masks, optimizer, scheduler, device, epochs=retrain_epochs)

        test_acc = evaluate(sparse_model, test_loader, device)
        kept = sum(m.sum().item() for m in current_masks.values())
        total = sum(m.numel() for m in current_masks.values())
        sparsity = 1 - kept / total
        print(f"[Round {r+1}] Acc={test_acc:.4f}, Sparsity={sparsity:.3f}")

    return sparse_model, current_masks, base_acc
```

- 실행 가이드  
  - 소규모부터 시작: total_rounds=3, prune_per_round=0.2처럼 완만한 프루닝을 권장합니다.[5][7]
  - 리와인드 구현: 교육용 코드는 W0에 리와인드합니다. 실제로는 “초반 k 에폭 체크포인트”에 리와인드하면 수렴이 안정적입니다.[7][5]
  - 평가: 각 라운드 sparsity와 정확도를 모니터링하여, 정확도 급락 지점을 회피합니다.[5][7]

## 학습 전 프루닝(SNIP) 예시 스니펫
아래는 학습 전 한 번의 saliency 계산으로 마스크를 만드는 핵심 아이디어입니다. 실제 구현에서는 소수 배치로 손실과 $dL/dW$ 를 얻고, $|∂L/∂W ⊙ W|$ 를 saliency로 사용합니다[8][9].

```python
def snip_saliency(model, loss_fn, data_loader, device, num_batches=1):
    model.to(device)
    model.train()
    grads = {}
    # 1) 모든 가중치에 대한 grad 수집
    batch_count = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.dim() > 1:
                if name not in grads:
                    grads[name] = p.grad.detach().clone()
                else:
                    grads[name] += p.grad.detach()
        batch_count += 1
        if batch_count >= num_batches:
            break
    # 2) saliency = |grad * weight|
    saliency = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and p.dim() > 1:
                saliency[name] = (grads[name] * p).abs()
    return saliency

def snip_prune_from_saliency(model, saliency, prune_ratio=0.5):
    # 전역 임계값 기반 마스크 생성
    all_scores = torch.cat([s.view(-1) for s in saliency.values()])
    k = int(prune_ratio * all_scores.numel())
    threshold = torch.kthvalue(all_scores, k).values
    masks = {}
    for name, s in saliency.items():
        masks[name] = (s > threshold)
    apply_masks(model, masks)
    return masks
```

- 워크플로우: 초기화→SNIP saliency 추정(1~N 배치)→마스크 생성→마스크 고정하고 표준 학습을 진행합니다.[9][8]
- 장점/주의: 사전 학습 없이 프루닝 가능하지만, 계층 간 상호작용을 충분히 반영하지 못해 매우 고희소 영역에서는 성능 저하가 나타날 수 있습니다.[8][9]

## 실험 설계 체크리스트
- 기준선 확보: 동일 모델·데이터로 dense 학습 성능을 먼저 확보합니다.[3][2]
- 희소성 스케줄: 프루닝 비율을 점증적으로 늘리며 정확도-희소성 트레이드오프 곡선을 수집합니다.[7][5]
- 재현성: 초기화 시드, 리와인드 시점, 학습률 스케줄을 기록합니다. LTH는 초기화와 초반 잡음에 민감합니다.[3][5]

## 더 깊게 공부하기
- 원 논문: Frankle & Carbin(2018/2019) — IMP, winning ticket 개념 확립.[2][5]
- 사전 프루닝: SNIP — 연결 민감도 기반 single-shot.[9][8]
- 서베이: 2024년 LTH 종합 정리 — 이론·응용·오픈 이슈·벤치마크 논의.[10][1]
- 이론·해설 글: 실무 관점 요약과 논쟁·재현성 이슈를 다룬 블로그·리뷰 글도 유용합니다.[4][3]

## 마무리 팁
- “작게 시작, 천천히 자르기”: 소형 모델·데이터셋에서 IMP 과정을 익힌 뒤 규모를 확장하세요.[5][7]
- “리와인드로 안정화”: 완전 초기화 대신 초반 체크포인트로 되돌리면 성능 유지가 쉬워집니다.[7][5]
- “혼합 전략”: SNIP으로 초회 마스크를 만들고, 짧은 IMP 라운드로 보정하는 하이브리드가 실용적입니다.[8][5]

참고 자료  
- The Lottery Ticket Hypothesis 원문 및 공식 IMP 알고리즘 설명.[2][5]
- SNIP: 학습 전 single-shot 프루닝의 핵심 아이디어와 구현 포인트.[9][8]
- 최근 서베이: LTH의 현재 위치와 남은 연구 과제, 실험 표준화 논의.[10][1]

[1](https://arxiv.org/html/2403.04861v1)
[2](https://arxiv.org/abs/1803.03635)
[3](https://cameronrwolfe.me/blog/lottery-ticket-hypothesis)
[4](https://www.lesswrong.com/posts/Z7R6jFjce3J2Ryj44/exploring-the-lottery-ticket-hypothesis)
[5](https://arxiv.org/pdf/1903.01611.pdf)
[6](http://www.theparticle.com/cs/bc/dsci/1803.03635.pdf)
[7](https://arxiv.org/pdf/2109.09670.pdf)
[8](https://arxiv.org/abs/1810.02340)
[9](https://arxiv.org/pdf/1810.02340.pdf)
[10](https://arxiv.org/abs/2403.04861)
[11](https://ui.adsabs.harvard.edu/abs/2020arXiv200600896V/abstract)
[12](https://simpling.tistory.com/m/58)
[13](https://www.semanticscholar.org/paper/c23173e93f1db79a422e2af881a40afb96b8cb92)
[14](https://www.semanticscholar.org/paper/21937ecd9d66567184b83eca3d3e09eb4e6fbd60)
[15](https://arxiv.org/abs/2401.10484)
[16](https://arxiv.org/abs/2410.14754)
[17](https://ieeexplore.ieee.org/document/10569031/)
[18](https://ieeexplore.ieee.org/document/10625908/)
[19](https://iopscience.iop.org/article/10.35848/1347-4065/ad2656)
[20](https://www.semanticscholar.org/paper/2028710190373ef893e3055c9113e04274a152d7)
[21](https://www.semanticscholar.org/paper/e71aed7a0680c8fc09733f1dcd0cd3f6bb9cb7aa)
[22](https://arxiv.org/pdf/2002.00585.pdf)
[23](https://arxiv.org/pdf/2403.04861.pdf)
[24](http://arxiv.org/pdf/2203.04248.pdf)
[25](http://arxiv.org/pdf/2206.04270.pdf)
[26](https://arxiv.org/pdf/2305.12148.pdf)
[27](https://arxiv.org/html/2504.05357v1)
[28](https://arxiv.org/pdf/2006.07014.pdf)
[29](https://arxiv.org/abs/2207.07858v1)
[30](https://arxiv.org/pdf/2111.00162.pdf)
[31](https://arxiv.org/pdf/2111.11146.pdf)
[32](https://en.wikipedia.org/wiki/Lottery_ticket_hypothesis)
[33](https://www.themoonlight.io/ko/review/insights-into-the-lottery-ticket-hypothesis-and-iterative-magnitude-pruning)
[34](https://proceedings.mlr.press/v119/malach20a/malach20a.pdf)
[35](https://proceedings.neurips.cc/paper/2021/file/15f99f2165aa8c86c9dface16fefd281-Paper.pdf)
[36](https://jiheek.tistory.com/24)
[37](https://xxxxxxxxxxxxxxxxx.tistory.com/entry/ICLR-2019-SNIP-Single-shot-Network-Pruning-based-on-Connection-Sensitivity)
[38](https://openreview.net/forum?id=9ZUz4M55Up)
[39](https://aigs.unist.ac.kr/filebox/item/1917192674_3f038f79_25433-Article+Text-29496-1-2-20230626.pdf)
[40](https://www.semanticscholar.org/paper/SNIP:-Single-shot-Network-Pruning-based-on-Lee-Ajanthan/cf440ccce4a7a8681e238b4f26d5b95109add55d)
[41](https://proceedings.mlr.press/v202/lange23a/lange23a.pdf)


https://simpling.tistory.com/m/58
