# ZeRO-Offload: Democratizing Billion-Scale Model Training

## 1. 핵심 주장 및 주요 기여  
ZeRO-Offload는 **단일 GPU+CPU 시스템**에서 모델 크기를 10배 이상 확장하면서도 **계산 효율성을 유지**하는 새로운 이종 하드웨어 기반 학습 기술이다.  
- **핵심 주장**: GPU 메모리 병목을 CPU 메모리·컴퓨팅으로 최적의 방식으로 오프로드함으로써, 추가적인 모델 변경 없이 최대 13B 파라미터 모델을 단일 V100 GPU에서 학습 가능하다.[1]
- **주요 기여**:  
  1. GPU-CPU 간 통신량 최소화·CPU 연산량 저감·GPU 메모리 절감의 **유일 최적 오프로드 전략** 제시.  
  2. ZeRO 기반 데이터 병렬성과 결합해 **128 GPUs까지 선형 확장** 가능.  
  3. 고성능 CPU Adam 구현(최대 6× 가속) 및 **One-step Delayed Update**로 CPU 오버헤드 은닉.  
  4. PyTorch DeepSpeed 라이브러리로 **무수정 사용** 가능한 공개 구현.

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
- **GPU 메모리 제한**: 대형 Transformer 모델(10B+) 학습 시 파라미터·그라디언트·옵티마이저 상태 메모리가 GPU에 수용 불가.  
- 기존 분산 기법(모델·파이프라인·ZeRO)은 다수 GPU 요구, 소수 사용자만 활용 가능.

### 2.2 제안 방법  
#### 2.2.1 데이터-플로우 그래프 분할  
모델 상태 및 연산을 아래 세 그룹으로 분류하여 이종 디바이스에 할당:  
- GPU: 순전파·역전파 연산 $$O(MB)$$ 및 fp16 파라미터  
- CPU: fp16 그라디언트 $$g_{16}$$, fp32 파라미터·모멘텀·분산 $$p_{32},m_{32},v_{32}$$ 및 옵티마이저 업데이트 $$O(M)$$  

#### 2.2.2 최소 통신량 및 최대 메모리 절감  
- **최소 통신량**: GPU–CPU 간 교환량을 $$4M$$ bytes로 최적화  
- **메모리 절감**: GPU 메모리 사용량을 baseline 대비 최대 **8×** 축소  

#### 2.2.3 옵티마이저 업데이트  
- CPU에서 다음 수식을 통해 Adam 업데이트 수행:  

$$
    m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad
    v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
  $$  

```math
    \hat{m}_t = \frac{m_t}{1-\beta_1^t},\quad
    \hat{v}_t = \frac{v_t}{1-\beta_2^t},\quad
    p_t = p_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
```

- **One-step Delayed Update**: CPU 연산과 GPU 연산을 겹치도록 지연 업데이트를 실시해 작은 배치에서도 CPU 병목 해소.

### 2.3 모델 구조  
- 기존 Transformer 구조 유지  
- **추가 프레임워크 변경 불필요**: DeepSpeed 라이브러리 플래그 설정만으로 활성화 가능

### 2.4 성능 향상  
- **단일 V100 GPU**: 1.2B → 13B 파라미터 학습(10×)  
- **연산 효율**: 30 TFLOPS → 40 TFLOPS(10B 모델)  
- **다중 GPU(128장)**: near-linear 스케일업, 30 TFLOPS/GPU 유지  
- **CPU-Adam**: PyTorch 대비 최대 6.4× 속도(1B 파라미터 기준)  
- **One-step Delayed Update**: 12–59% 추가 성능 향상  

### 2.5 한계  
- **CPU 메모리·대역폭 의존**: 대규모 클러스터 환경에서 CPU 리소스 분산 고려 필요  
- **소규모 배치**에서만 CPU 병목 가능성 잔존  
- 활성화(offload) 전략은 *모델 상태*에 국한, *activation memory* 최적화는 별도 기법과 병행 필요

## 3. 일반화 성능 향상 관점  
- 파라미터 수 증가는 **학습 곡선의 기울기 완만화** 및 표현력 증가를 통해 일반화 성능 개선에 직결.  
- ZeRO-Offload로 **더 큰 모델**을 손쉽게 학습함으로써, *overfitting* 제어 및 *few-shot* 학습 능력 강화 가능성.  
- **Delayed Update**는 파라미터 스텝 스테일리티를 1단계 허용하나, 초기 안정화 후에는 **수렴 저해 없음**을 검증.[1]

## 4. 향후 연구에의 영향 및 고려 사항  
- **범용성 확장**: GPU-CPU 외 FPGA·ASIC 환경, 또는 다중 계층 메모리(예: NVMe) 오프로드 적용 연구  
- **활성화 체크포인팅 결합**: ZeRO-Offload와 함께 activation memory 최적화 병행으로 추가 메모리 이득  
- **통신 인프라 최적화**: 고대역폭 상호연결(In-Network Compute) 활용한 GPU-CPU 통신 병목 해소  
- **일반화 이론 연구**: 대규모 모델 학습이 실제 downstream task 일반화 성능에 미치는 영향 정량 분석 연구 필요  	

ZeRO-Offload는 대형 모델 학습 민주화를 위한 **이종 하드웨어 분할** 전략을 제시하며, 향후 초거대 AI 연구의 기반 기술로 자리매김할 전망이다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/63435e7b-d881-44ff-a9ac-7587fe416f85/2101.06840v1.pdf)
