# Techniques for training large neural networks

핵심은 대규모 모델 학습을 위해 연산을 여러 GPU로 분해하고, 통신·메모리 병목을 완화하는 **데이터 병렬**, **파이프라인 병렬**, **텐서 병렬**, **MoE**를 적재적소에 조합하는 것입니다.  
또한 활성값 체크포인팅, 혼합 정밀, 오프로딩, ZeRO, 메모리 효율 옵티마이저, 통신 압축 등 실전 팁도 함께 정리합니다.[1][2]

## 큰 그림
대규모 모델 학습은 순전파·역전파·옵티마이저 업데이트라는 반복 과정 전체를 여러 장치에 분산하는 문제입니다.  
모델·배치가 단일 GPU 메모리를 초과하면, 계산을 데이터/레이어/연산 차원으로 나눠 통신과 유휴 시간(일반적으로 컴퓨터 시스템, 장비, 또는 사람이 사용 가능한 상태이지만 실제적인 작업이 없어 낭비되는 시간)을 줄여야 합니다.  
이 글은 각 병렬화 패턴의 개념, 장단점, 실제 코드 적용 포인트를 짧은 문장으로 정리합니다.[1]

## 데이터 병렬
데이터 병렬은 동일한 모델 복제본을 여러 GPU에 배치하고, 배치를 샤딩(대규모 데이터를 **샤드(Shard)** 라고 하는 작은 조각으로 분할하여, 각 샤드를 서로 다른 데이터베이스 서버(컴퓨터)에 분산 저장함으로써 부하 분산과 성능 향상, 그리고 확장성 확보을 가능하게 하는 기술)하여 독립적으로 그라디언트를 계산한 뒤 동기 All-Reduce로 평균화합니다.  
동기식(여러 구성 요소가 동시에 맞춰져서 함께 작동하는 방식. AI 학습에서 동기식은 병렬 처리 단계들이 동시에 진행돼야 함을 뜻하며, 모델이 각 GPU에 “적어도 한 번” 올라가야 하는 제한이 있습니다.)은 학습 효율이 높고, 구현이 단순하며, 단일 GPU에 모델이 “적어도 한 번”은 올라가야 한다는 제약이 있습니다.  
통신 오버헤드는 파라미터 크기에 비례하므로, 혼합 정밀·그라디언트 압축이 실전에서 중요합니다.[1]

- 장점: 구현 단순, 확장 용이, 학습 안정적입니다.[1]
- 단점: 거대 모델은 단일 복제본조차 메모리에 안 들어갈 수 있습니다.[1]
- 팁: AMP(FP16/BF16)와 Loss Scaling, 통신 백엔드 최적화(NCCL), DDP의 gradient bucketing을 사용합니다.[1]

### All-Reduce
All-Reduce는 분산 환경에서 여러 노드 또는 GPU가 계산한 값을 모두 합산한 후, 그 결과를 모든 노드에 전달하는 기법입니다.  
이때 평균화를 하려면 합산된 값을 노드 수로 나누어 각 노드가 동일한 평균 값을 가지도록 합니다.

분산 딥러닝에서 주로 그래디언트(기울기)를 동기화할 때 All-Reduce를 사용하여 그래디언트를 합산한 후, 노드 수로 나누어 평균 그래디언트를 구합니다.  
이렇게 해야 모델 파라미터가 올바르게 업데이트되며, 각 GPU가 동일한 평균값을 사용해 학습하게 됩니다.  
NVIDIA의 NCCL 라이브러리 및 PyTorch, TensorFlow 등도 All-Reduce로 평균 계산을 지원합니다.

#### NCCL, NVIDIA Collective Communication Library
NCCL은 NVIDIA가 개발한 멀티 GPU 및 멀티 노드 환경에서 고성능 통신을 지원하는 라이브러리로, GPU 간의 집단 통신(collective communication) 연산들을 최적화하여 매우 낮은 지연시간과 높은 대역폭을 제공합니다.  
주요 기능은 all-gather, all-reduce, broadcast, reduce, reduce-scatter 같은 집단 연산과 point-to-point 통신이 포함됩니다.

### PyTorch 예시: 기본 DDP
- 핵심: 각 프로세스가 1 GPU를 전담. 모델을 DistributedDataParallel로 감쌉니다.[1]
- 효과: 동기 그라디언트 평균(All-Reduce)로 동일 업데이트를 보장합니다.[1]

```python
# torchrun --nproc_per_node=NUM_GPUS train_ddp.py
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP # 모델을 DDP 래퍼로 감싸 동기식 All-Reduce를 자동 처리합니다.
from torch.utils.data import DataLoader, DistributedSampler # 샘플러가 데이터셋을 프로세스 단위로 샤딩해 중복 없이 배분합니다.

def setup():
    dist.init_process_group(backend="nccl") # 프로세스 그룹 생성, GPU 간 통신 백엔드는 NCCL을 사용합니다.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"])) # 현재 프로세스를 로컬 GPU 인덱스에 바인딩합니다.

def cleanup(): # 학습 종료 시 자원 정리를 수행합니다.
    dist.destroy_process_group() 

def main():
    setup() # 분산 초기화를 먼저 수행합니다.
    local_rank = int(os.environ["LOCAL_RANK"]) # 이 프로세스가 담당하는 GPU 번호입니다.
    device = torch.device(f"cuda:{local_rank}") # 텐서를 올릴 디바이스를 결정합니다.

    model = MyModel().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank) # DDP로 감싸서 역전파 시 그라디언트 동기화를 자동화합니다.

    dataset = MyDataset()
    sampler = DistributedSampler(dataset, shuffle=True) # 각 프로세스별 데이터 샤딩 및 셔플 동기화를 담당합니다.
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler() # 혼합 정밀(AMP) 시 손실 스케일링을 담당합니다.
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True) # 비동기 전송으로 호스트-디바이스 복사를 숨깁니다.
            optimizer.zero_grad(set_to_none=True) # 그라디언트 버퍼를 None으로 두어 메모리 할당을 최적화합니다.
            with torch.cuda.amp.autocast(): # FP16/BF16로 연산을 낮춰 속도·메모리를 절약합니다.
                loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    cleanup()

if __name__ == "__main__":
    main()
```
참고: torchrun은 LOCAL_RANK, RANK, WORLD_SIZE를 자동 주입합니다. 

## 파이프라인 병렬
연속된 레이어를 여러 GPU에 수직 분할하여, 각 장치가 모델의 “일부 레이어만” 보유하도록 합니다. 마이크로배치로 파이프를 채워 버블(유휴 시간)을 최소화하는 것이 핵심입니다. GPipe는 동기식 집계로 안정적이며, PipeDream은 전방/역방향을 교대로 스케줄링하여 처리량을 높입니다.[3][4][1]

- 장점: 단일 장치 메모리 한계를 넘어 모델 자체를 확장합니다.[4][3]
- 단점: 스테이지 경계 통신, 버블 관리, 배치-의존 연산(BatchNorm) 처리 복잡성입니다.[3]
- 팁: 스테이지 간 균형 잡힌 FLOPs 분할, 마이크로배치 수를 늘려 버블 축소, 리마테리얼라이제이션((rematerialization)"은 계산 과정 중에 저장하지 않고 다시 계산하여 메모리 사용을 줄이는 기술입니다. 예를 들어, 중간 결과를 저장하는 대신 필요할 때마다 다시 계산함으로써 메모리 절약과 처리 효율을 도모합니다.)과 병행합니다.[5][3]

### PyTorch 예시: 파이프라인 데모
- 핵심: torch.distributed.pipeline(실전은 torchgpipe, DeepSpeed Pipeline 등 프레임워크 사용 권장)으로 Sequential을 stage로 쪼갭니다.[3]

```python
# 개념 데모: 실제 실전은 torchgpipe/DeepSpeed Pipeline을 권장
import torch.nn as nn

stage0 = nn.Sequential(nn.Linear(4096, 4096), nn.GELU()).to("cuda:0")
stage1 = nn.Sequential(nn.Linear(4096, 4096), nn.GELU()).to("cuda:1")
stage2 = nn.Sequential(nn.Linear(4096, 4096), nn.GELU()).to("cuda:2")
stage3 = nn.Sequential(nn.Linear(4096, vocab)).to("cuda:3")

# GPipe류를 사용하면:
# from torchgpipe import GPipe # GPipe 래퍼는 마이크로배치로 파이프를 채워 버블을 최소화합니다.
# model = nn.Sequential(stage0, stage1, stage2, stage3)
# model = GPipe(model, balance=[...], chunks=micro_batches)

# PipeDream/DeepSpeed Pipeline은 스케줄링이 상이합니다.
```

### GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism
GPipe는 구글 브레인에서 개발한 파이프라인 병렬 처리 라이브러리로, 메모리 소모가 큰 대규모 신경망 모델을 효율적으로 학습할 수 있게 해줍니다.  
이 라이브러리는 모델을 여러 파티션으로 나누어 각 파티션을 다른 장치(GPU 또는 TPU)에 분배하고, 미니배치를 작은 마이크로배치로 나누어 병렬 처리 효율을 극대화합니다.  
또한, 역전파 시 필요할 때만 중간 계산 값을 재연산하는 체크포인팅 기법으로 메모리 사용을 줄입니다.

## 텐서 병렬
레이어 내 대형 행렬곱을 열/행으로 샤딩하여 여러 GPU가 한 연산을 협업 처리합니다.  
Megatron-LM은 Self-Attention과 MLP에 컬럼/로우 패럴렐 선형층을 정의해 All-Reduce/All-Gather 패턴으로 결과를 결합합니다.  
시퀀스 병렬은 토큰 축을 분할해 활성 메모리를 낮추는 보완 기법입니다.[6][7][8][9][1]

- 장점: 레이어 내부 병목을 분해하여 단일 레이어 파라미터/활성 메모리 압박을 줄입니다.[7][6]
- 단점: 통신 패턴이 복잡하고, 토폴로지 의존성이 큽니다(고속 링크 필요).[6][7]
- 팁: 텐서 병렬 그룹 크기를 NVLink/IB 토폴로지에 맞추고, 컬럼→로우 패턴으로 All-Reduce 수를 최소화합니다.[7][6]

### All-Gather
All-Gather는 여러 프로세스 또는 장치에서 각각의 데이터를 모아서 모든 프로세스가 모인 전체 데이터를 받는 집단 통신 연산입니다.

구체적으로, 각 랭크가 N개의 값을 갖고 있을 때, k개의 랭크 모두의 데이터를 모아서 크기가 k*N인 버퍼에 순서대로 정리한 후 이 결과를 모든 장치에 뿌려줍니다. 즉, 모든 참여자가 전체 데이터를 공유하게 됩니다.

이를 요약하면:

- 각 프로세스가 가진 데이터를 모아서(모든 랭크의 데이터를 모두 모음)
- 모든 프로세스가 이 결과 데이터를 갖게 하는 연산
- MPI에서는 MPI_Allgather, NVIDIA NCCL에서는 AllGather로 불림
- AllReduce와 달리 단순히 데이터를 모아서 분배하기 때문에 모든 데이터의 집계를 수행하지 않음
- Jax 라이브러리에서는 해당 동작을 빠르게 수행하는 all_gather 함수 제공

이 연산은 분산 학습, 병렬 처리에서 중간 결과를 모아서 모두가 동일한 데이터를 가질 때 주로 활용됩니다.

### NVLink/IB
NVLink는 같은 서버 내 GPU 간 초고속 통신을 위한 연결 기술로, 주로 단일 노드 내 GPU를 직접 연결하는 데 사용됩니다.  
반면, InfiniBand는 여러 서버(노드) 간 고대역폭, 저지연 통신을 지원하는 클러스터 간 네트워크 기술입니다.  
즉, NVLink는 ‘서버 내’ GPU 간 통신용, InfiniBand는 ‘서버 간’ 통신용입니다.

NVLink 토폴로지는 다수 GPU를 고속 링크로 직접 연결하거나 NVSwitch라는 고성능 스위칭 장치를 통해 다수 GPU를 스위칭 방식으로 연결합니다.  
예를 들어, NVIDIA의 NVSwitch 아키텍처를 활용하면 최대 수백에서 천여 개 GPU를 연결하는 대규모 토폴로지도 가능하며, GPU 간 직접 통신과 네트워크 집계 기능을 지원해 트레이닝 성능을 높입니다.

InfiniBand 토폴로지는 저지연, 고대역폭의 네트워크 스위치를 통해 서버 간 연결을 구성하며, 특히 대규모 AI 클러스터에서 중요합니다.  
최근에는 GPU가 직접 NIC(Network Interface Card)와 PCIe 스위치를 통해 통신하는 ‘GPUDirect RDMA’ 기술을 활용해 InfiniBand 네트워크의 효율성을 극대화합니다.  
이로써 노드 간 GPU 간 통신 지연을 줄이고 대역폭을 높여 대규모 분산 학습에 적합한 네트워크를 구성합니다.

### PyTorch 예시: Megatron 스타일 선형 샤딩
- 핵심: ColumnParallelLinear와 RowParallelLinear를 조합하여 MLP 두 선형층을 분산합니다.[6][7]

```python
# 개념적 의사코드: 실전은 Megatron-LM/DeepSpeed-TensorParallel 사용 권장
class ColumnParallelLinear(nn.Module): # 출력 차원을 샤딩하는 선형층 정의입니다.
    def __init__(self, in_f, out_f, tp_rank, tp_world): # 텐서 병렬 그룹 내 랭크(장치)/월드 크기를 받습니다.
        super().__init__()
        shard = out_f // tp_world # 출력 피처를 장치 수로 균등 분할합니다.
        self.weight = nn.Parameter(torch.empty(in_f, shard, device=f"cuda:{tp_rank}")) # 이 랭크가 담당하는 열 조각 가중치입니다.
        self.bias = nn.Parameter(torch.zeros(shard, device=f"cuda:{tp_rank}"))
    def forward(self, x):  # x: [B, in_f], broadcasted
        y_local = x @ self.weight + self.bias   # [B, shard] , 이 랭크의 부분 출력입니다.
        return y_local  # 각 랭크 보유, 필요 시 all-gather

class RowParallelLinear(nn.Module): # 입력 차원을 샤딩하는 선형층 정의입니다.
    def __init__(self, in_f, out_f, tp_rank, tp_world):
        super().__init__()
        shard = in_f // tp_world
        self.weight = nn.Parameter(torch.empty(shard, out_f, device=f"cuda:{tp_rank}"))
        self.bias = nn.Parameter(torch.zeros(out_f, device=f"cuda:{tp_rank}"))
    def forward(self, x_sharded):  # [B, shard] 입력이 랭크별로 분할
        y_partial = x_sharded @ self.weight      # [B, out_f]
        y = torch.distributed.all_reduce(y_partial, op=torch.distributed.ReduceOp.SUM) # 부분 결과를 합산해 전체 출력을 만듭니다.
        return y + self.bias # 최종 바이어스를 더합니다(실전은 바이어스 처리도 병렬 설계에 맞춰 조정).
```

## Mixture-of-Experts(MoE)
게이트가 입력별로 소수 전문가만 선택하여 계산하는 “희소” 구조입니다. 동일 FLOPs에서 파라미터 수를 크게 늘릴 수 있어, 거대 모델 용량을 확보하면서 비용을 억제합니다. GShard, Switch Transformer는 TPU/GPU 다중 장치에 전문가를 분산 배치하여 수천억~수조 파라미터까지 확장했습니다.[1]

- 장점: **희소 활성화**로 대규모 파라미터 용량 확보, 장치 간 전문가 분산이 자연스럽습니다.[1]
- 단점: 라우팅 부하 불균형, 토큰 드롭, 통신 비용, 정규화/균형 손실 설계 이슈가 있습니다.[1]
- 팁: Top-1/Top-2 게이팅, Load-Balancing Loss, Capacity Factor, 전문가 정렬/샤딩을 활용합니다.[1]

### GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding
GShard는 구글이 개발한 대규모 신경망 모델을 효율적으로 확장하기 위한 시스템으로, 조건부 계산(Conditional Computation)과 자동 샤딩(Automatic Sharding)을 통해 6000억 개 이상의 파라미터를 가진 초거대 모델을 효과적으로 학습할 수 있게 해줍니다.  
특히 희소 전문가(Mixture of Experts, MoE) 구조와 토큰의 하드 라우팅(hard routing)을 활용하여 계산 효율성을 극대화합니다.

주요 특징은 다음과 같습니다:

- 조건부 계산 방식: 각 입력 토큰이 하드 라우팅되어, 오직 일부 전문가(experts)만 처리에 참여함으로써 전체 연산 비용을 크게 줄입니다.
- 자동 샤딩: XLA 컴파일러 확장과 경량 주석(annotation) API를 통해 수천 개의 TPU 장치에 모델을 자동으로 분산시켜 병렬 처리합니다.
- 대규모 다국어 번역 모델: 2048개의 TPU v3 코어에서 4일만에 100개 언어의 번역 모델을 학습시켜 이전 기술보다 현저히 높은 품질을 달성했습니다.
- 토큰별 top-2 전문가 선택: 각 토큰은 상위 2개의 전문가만 선택해 처리하며, 이는 기존 top-4 또는 top-1 전문가 선택 방식보다 균형 잡힌 효율성과 성능을 제공합니다.

이 시스템은 거대 신경망의 메모리 제한, 계산 비용, 병렬화 어려움의 문제를 효과적으로 극복하면서 동시에 프로그래밍 편의성도 높인 것이 특징입니다.

#### XLA(Accelerated Linear Algebra)
XLA(Accelerated Linear Algebra) 컴파일러는 OpenXLA 프로젝트에서 개발한 오픈 소스 머신러닝 컴파일러로, 계산 그래프를 저수준에서 최적화해 TensorFlow, PyTorch, JAX 같은 프레임워크의 모델 성능을 향상시키는 역할을 합니다.

### Top-1/Top-2 게이팅
Top-1/Top-2 게이팅은 Mixture of Experts(MoE) 모델에서 토큰을 전문가(Expert)에게 할당하는 라우팅 방식입니다.

- Top-1 게이팅은 라우터 스코어가 가장 높은 단 한 명의 Expert에게 토큰을 할당합니다.
- Top-2 게이팅은 스코어가 높은 상위 2개의 Expert에게 토큰을 할당하며, 두 Expert의 결과를 가중합하여 다음 레이어로 넘깁니다.

이 방식은 입력 토큰이 분산되어 여러 Expert에게 처리됨으로써, 모델의 효율성과 성능을 높이고 일부 파라미터만 활성화하는 희소성(Sparsity)을 달성합니다. 특히 Mixtral 모델이 Top-2 게이팅 방법을 사용합니다.

### Load-Balancing Loss(LBL)
Load-Balancing Loss(LBL)은 Mixture-of-Experts(MoE) 모델 훈련 시 전문가(Experts)의 활용을 균형 있게 유지하기 위한 정규화 손실입니다.  
이 손실은 특정 전문가에게 지나치게 많은 토큰이 집중되는 것을 방지합니다.

LBL 수식은 다음과 같습니다:

```math
[\text{LBL} = \frac{1}{N_E} \sum_{i=1}^{N_E} f_i \cdot p_i
]
```

여기서

- $(N_E)$는 전문가의 총 개수,
- $(f_i)$는 i번째 전문가에 라우팅되는 토큰의 비율,
- $(p_i)$는 i번째 전문가에 할당된 전체 라우팅 확률(라우팅 확률은 네트워크에서 메시지나 데이터가 특정 노드나 경로를 통해 전달될 가능성을 수치적으로 나타낸 것입니다.)을 의미합니다.

이 손실을 최소화함으로써 전문가들이 분포된 토큰을 공평하게 처리하도록 유도하여, 일부 전문가가 과부하되거나 활용되지 않는 문제(Expert Collapse)를 방지합니다.

#### Expert Collapse
Expert Collapse, 특히 Sparse Mixture of Experts (SMoE) 모델에서 나타나는 현상으로, 여러 전문가(experts)가 거의 동일하거나 매우 유사한 표현(embedding)만 학습하는 문제를 의미합니다. 이로 인해 일부 전문가만이 크게 활용되고, 나머지 전문가들은 비효율적으로 사용되어 모델 전체 성능 저하가 발생합니다.

이 문제는 주로 라우팅 메커니즘이 입력 토큰들을 특정 전문가에 몰아주는 경향에서 비롯되며, 전문가 간 표현의 다양성이 감소하는 결과를 초래합니다. 대표적으로 전문가들이 같은 입력 패턴에 대해 비슷한 표현을 갖게 되어, 모델의 표현력이 떨어지고 불안정해집니다.

이를 해결하기 위해 최근 연구들은 다음과 같은 방법을 제안합니다:

SimSMoE: 전문가들 사이 표현의 유사도를 측정해(CKA 손실함수 사용), 유사도를 낮춤으로써 표현 붕괴를 완화합니다. 특히, 토큰 공유 빈도 기반으로 붕괴 가능성이 있는 전문가를 찾아내어 세밀하게 조정합니다.

저차원 임베딩 공간 투영 및 개선된 라우팅 알고리즘을 통해 전문가 클러스터가 명확히 분리되도록 하여 표현 붕괴를 완화하는 방식도 연구되고 있습니다.

최근 연구에서는 이 보조 손실 없이도 효율적인 균형을 유지하는 Loss-Free Balancing 전략도 제안되어, 모델 성능 저하 없이 더 나은 로드 밸런스를 달성하는 결과를 보여주고 있습니다.

### Capacity Factor
MoE(Mixture of Experts)에서 **Capacity Factor(용량 계수)**는 주로 모델 내 전문가(experts)의 처리 용량을 의미하며, 이는 모델 효율성과 연산 비용에 직결됩니다.  
Capacity Factor는 선택된 전문가 수, 각 전문가의 hidden size, 그리고 게이팅 네트워크가 할당하는 토큰 수에 따라 결정되며, 이를 조정하면 비용(연산량과 메모리 요구량)에 영향을 줍니다.

### PyTorch 예시: 간단 Top-2 MoE 블록
- 핵심: 게이트가 전문가 로짓에서 상위 k를 선택. 토큰을 라우팅, 각 전문가 MLP 통과, 병합합니다.[1]

```python
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
    def forward(self, x):
        return self.net(x)

class Top2Gating(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.w_g = nn.Linear(d_model, n_experts, bias=False)
    def forward(self, x):  # x: [B, T, D]
        logits = self.w_g(x)  # [B, T, E]
        top2_val, top2_idx = logits.topk(k=2, dim=-1)
        gate = top2_val.softmax(dim=-1)  # [B, T, 2]
        return top2_idx, gate

class MoeLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        self.gate = Top2Gating(d_model, n_experts)
    def forward(self, x):
        idx, gate = self.gate(x)  # idx: [B,T,2]
        # 라우팅(간단 데모: for 루프; 실전은 패킹/언패킹+scatter/gather)
        y = torch.zeros_like(x)
        for k in range(2):
            m = idx[..., k]  # [B,T]
            # 마스크로 전문가별 토큰 선택 및 처리 (실전은 정렬/패킹으로 병렬화)
            for e in range(len(self.experts)):
                sel = (m == e)
                if sel.any():
                    xe = x[sel]
                    ye = self.experts[e](xe)
                    y[sel] += gate[..., k][sel].unsqueeze(-1) * ye
        return y
```


## 메모리/통신 최적화
- 활성값 체크포인팅: 역전파 시 필요한 일부 활성만 저장하고, 나머지는 재계산해 메모리를 절약합니다. 파이프라인과 함께 버블을 줄이는 데 쓰입니다.[3][1]
- 혼합 정밀(AMP): FP16/BF16로 연산·저장, FP32 master weight/accumulation, Loss Scaling으로 안정성 확보합니다. OpenAI 블로그 축약본에서도 대표 팁으로 다룹니다.[1]
- 오프로딩/ZeRO: CPU/디스크/다른 장치로 파라미터·그라디언트·옵티마 상태를 분산·지연 물질화하여 메모리 사용량을 줄입니다.[1]
- 압축: 통신 전 그라디언트/활성을 양자화·스파스화하여 대역폭을 절약합니다.[1]

### FP16/BF16
FP16과 BF16은 16비트 부동소수점(floating-point) 형식으로, 모두 연산과 저장에 메모리 절약과 속도 향상을 위해 사용됩니다. FP16은 1비트 부호, 5비트 지수, 10비트 가수로 구성되어 표현 범위와 정밀도가 제한되는 반면, BF16은 1비트 부호, 8비트 지수, 7비트 가수로 지수 범위가 FP16보다 넓어 더 안정적입니다.

이 두 형식 모두 FP32(32비트 부동소수점)보다 메모리 사용량을 절반가량 줄일 수 있어 대규모 모델 학습과 추론에 유리하나, 정밀도 손실 가능성이 존재합니다. 따라서 보통 학습 시에는 FP32를 기본으로 하되, 추론이나 일부 연산에서 FP16/BF16을 사용해 연산 속도를 높이고 메모리를 절약합니다.

특히 BF16은 인공지능 하드웨어(예: 인텔 최신 프로세서)에서 많이 지원하며, FP16보다 지수 범위가 넓어 NaN 및 overflow 문제가 적지만, 가수 비트가 적어 세밀한 정밀도는 낮아집니다.

### FP32 master weight/accumulation
FP32 master weight는 혼합 정밀도 학습(Mixed Precision Training)에서 반드시 FP32(32비트 부동소수점) 형태로 유지되는 가중치의 주 복사본을 말합니다.  
이는 FP16(16비트)로 가중치 업데이트를 하면 갱신값이 너무 작아 손실될 수 있기 때문에, 정확한 누적 계산(가중치 누적, accumulation)을 위해 FP32로 업데이트를 수행하고 FP16으로 변환하여 계산에 활용하는 방식입니다.

구체적으로, FP32 마스터 가중치는 매 학습 반복(iteration)마다 FP16 가중치로 변환되어 네트워크 계산에 사용되고, 업데이트 시에는 FP16 기울기를 FP32로 변환하여 마스터 가중치에 더해지는 구조입니다.  
이렇게 하면 FP16 표현 한계로 인해 발생할 수 있는 업데이트 손실 문제를 방지하여 FP32와 유사한 학습 정확도를 유지할 수 있습니다.

또한, 손실 경사 값이 너무 작아 FP16에서 0으로 사라지는 현상을 막기 위해 Loss Scaling 기법을 병행해 작은 기울기를 스케일링했다가 업데이트 시 다시 역스케일링합니다.  

여기서 Loss scaling은 손실 값에 큰 스케일링 팩터를 곱해 주어 FP16에서 너무 작은 그래디언트가 0으로 사라지는 문제를 해결하며, 역전파 후에는 이 스케일을 다시 나누어 원래 크기로 복원합니다.

이로써 메모리 사용량은 FP32 대비 절반 수준으로 줄이고, 계산 속도는 높이면서도 성능을 유지할 수 있습니다.

이러한 방식을 통해 혼합 정밀도 학습에서 FP16이 가지는 한계점(정밀도 손실) 문제를 해결합니다.

### PyTorch 예시: 체크포인팅 + AMP 조합
```python
import torch.utils.checkpoint as cp

class Block(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self, x):
        return cp.checkpoint(self.f, x)  # 활성값 메모리 절감

scaler = torch.cuda.amp.GradScaler()
for x, y in loader:
    with torch.cuda.amp.autocast():
        loss = model(x, y)
    scaler.scale(loss).backward() # 스케일된 역전파
    scaler.step(optimizer)
    scaler.update()
```

## 언제 무엇을 쓰나
- 단일 GPU에 “모델이 들어간다면” 데이터 병렬 + AMP부터 시작합니다.[1]
- 모델이 단일 GPU에 “안 들어가면” 파이프라인 병렬로 수직 분할을 고려합니다.[4][3]
- 레이어 내부 행렬곱이 병목(bottleneck)이면 텐서 병렬을 섞어 통신-계산 균형을 맞춥니다.[6][7]
- FLOPs 예산 내 파라미터 용량을 늘리고 싶다면 MoE를 채택합니다.[1]
- 실전에서는 데이터+파이프라인+텐서(3D 병렬) 조합이 일반적이며, GPipe·Megatron-LM 구현이 좋은 출발점입니다.[4][7][1]

## 참고 구현과 자료
- OpenAI가 정리한 병렬화/메모리 절약 기법의 개요는 Lilian Weng의 정리 글과 그 축약본에서 명확합니다.[2][1]
- GPipe: 마이크로배치 파이프라이닝, 버블 최소화, 안정적 동기식 업데이트 구조가 핵심입니다.[10][4][3]
- Megatron-LM: 컬럼·로우 병렬 선형층, Self-Attention 병렬화, 시퀀스 병렬과의 조합이 특징입니다.[8][9][7][6]

## 체크리스트
- 통신/계산 비율: All-Reduce/All-Gather 빈도와 텐서 크기를 모니터링합니다.[7][6]
- 토폴로지: NVLink/IB 링/트리 구성과 텐서 병렬 그룹 매핑을 일치시킵니다.[6]
- 버블: 파이프라인 스테이지 밸런싱, 마이크로배치 수 튜닝으로 유휴 시간을 줄입니다.[5][3]
- 안정성: AMP Loss Scaling, Grad Clipping, 동기식 업데이트로 수렴 품질을 지킵니다.[1]
- 메모리: 체크포인팅, ZeRO/오프로딩, 시퀀스 병렬로 피크 메모리를 낮춥니다.[8][1]

## 마무리
핵심 병렬화 축은 데이터·파이프라인·텐서이며, 필요에 따라 **MoE**로 희소성을 더해 용량을 확장합니다. 여기에 AMP, 체크포인팅, ZeRO, 통신 압축을 더하면, 연구 현장에서 거대 모델을 효율적으로 학습할 수 있습니다. 시작점으로는 DDP+AMP로 베이스라인을 만들고, 메모리 한계에서 파이프라인/텐서 병렬을 단계적으로 추가하는 전략이 권장됩니다.[3][7][1]

[1](https://lilianweng.github.io/posts/2021-09-25-train-large/)
[2](https://openai.com/index/more-on-dota-2/)
[3](https://fid3024.github.io/papers/2019%20-%20GPipe:%20Efficient%20Training%20of%20Giant%20Neural%20Networks%20using%20Pipeline%20Parallelism.pdf)
[4](https://arxiv.org/abs/1811.06965)
[5](https://www.sciencedirect.com/science/article/pii/S0167739X23001735)
[6](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-training/app_notes/nxd-training-tp-appnote.html)
[7](https://arxiv.org/pdf/1909.08053.pdf)
[8](https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf)
[9](https://arxiv.org/abs/1909.08053)
[10](https://dl.acm.org/doi/abs/10.5555/3454287.3454297)
[11](https://openai.com/index/techniques-for-training-large-neural-networks/)
[12](https://arxiv.org/abs/2406.02806)
[13](https://www.semanticscholar.org/paper/a54f732259be0f916a9e059b979a7852f95a61b5)
[14](https://ieeexplore.ieee.org/document/10268888/)
[15](https://www.aanda.org/10.1051/0004-6361/201935628)
[16](https://link.springer.com/10.1007/978-3-031-64892-2_11)
[17](https://ieeexplore.ieee.org/document/10702097/)
[18](https://www.semanticscholar.org/paper/dc5f19a7b8aab71baf08e4ae3024511807f866c6)
[19](https://muse.jhu.edu/article/906568)
[20](https://www.semanticscholar.org/paper/f8cac20d7ff28923139512bd58094833aefaa3ab)
[21](https://osf.io/aqc9n)
[22](https://arxiv.org/pdf/2110.03888.pdf)
[23](https://arxiv.org/html/2406.06811v2)
[24](https://arxiv.org/pdf/2303.10455.pdf)
[25](https://arxiv.org/html/2311.03233)
[26](http://arxiv.org/pdf/2406.06962.pdf)
[27](https://arxiv.org/pdf/1602.07868.pdf)
[28](https://arxiv.org/pdf/2308.01320.pdf)
[29](http://arxiv.org/pdf/2502.04066.pdf)
[30](https://academic.oup.com/nsr/advance-article-pdf/doi/10.1093/nsr/nwad300/53900418/nwad300.pdf)
[31](https://arxiv.org/pdf/2205.01068.pdf)
[32](https://www.reddit.com/r/mlscaling/comments/vbhfb2/techniques_for_training_large_neural_networks_by/)
[33](https://news.ycombinator.com/item?id=31682887)
[34](https://lilianweng.github.io)
[35](https://assemblyai.com/blog/how-to-train-large-deep-learning-models-as-a-startup)
[36](https://openai.com/index/how-ai-training-scales/)
[37](https://arize.com/blog/openai-on-rlhf/)
[38](https://arxiv.org/abs/2004.09910)
[39](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf)
[40](https://proceedings.neurips.cc/paper/2019/file/093f65e080a295f8076b1c5722a46aa2-Reviews.html)

# Reference
- https://openai.com/index/techniques-for-training-large-neural-networks/
