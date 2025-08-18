# Full Stack Optimization of Transformer Inference: a Survey

## 1. 핵심 주장 및 주요 기여
이 논문은 최근 Transformer 기반 모델의 추론 비용 증가 문제를 다루며, 모델 수준부터 하드웨어 가속기 설계, 연산 스케줄링, 모델 구조 탐색(NAS)에 이르는 풀스택 관점에서 Transformer 추론을 최적화하는 방법들을 종합적으로 분석·정리한다.  
주요 기여:
1. **병목 분석**: Transformer 인퍼런스의 연산·메모리·비선형 연산(Softmax, LayerNorm, GELU) 병목을 정량적·정성적으로 밝힘.  
2. **하드웨어 설계 지침**: CNN 전용 가속기와 달리, Transformer 전용 가속기 요구사항(비선형 연산 처리, 메모리 계층 구성 재조정 등)을 제시.  
3. **스케줄링·매핑 전략**: matmul 연산 스케줄링 복잡도, 비선형 연산 융합(fusion) 시 제약과 균형을 상세 분석.  
4. **NAS 적용 사례**: Gemmini 가속기 위에서 하드웨어 성능(Latency, EDP)과 언어 모델(perplexity)을 동시에 고려한 NAS 동시 최적화 사례를 제시.  

## 2. 해결하고자 하는 문제
- Transformer 모델 크기와 sequence 길이 증가로 인한 추론 연산량 및 메모리 대역폭 부담  
- Softmax, LayerNorm 등 비선형 연산이 전체 FLOPs 비중은 작지만, 메모리 접근·런타임 지연을 유발  
- 기존 CNN 가속기 설계·스케줄링 기법을 그대로 적용할 경우 극도로 낮은 하드웨어 활용도  
- CNN 대비 matmul 스케줄링 난이도는 낮아 보이나, 실제로는 유사한 검색 공간 크기

## 3. 제안 방법
1) **모델 분석**  
   - FLOPs/MOPs 산출 및 arithmetic intensity 계산  
   - CPU 프로파일링을 통한 연산별 latency 분해  
   - matmul(프로젝션 vs act-to-act), Softmax, LayerNorm 병목 도출

2) **하드웨어 설계**  
   - Gemmini 가속기 사례:  
     -  입출력 스크래치패드와 accumulator 크기 재조정 → 36% matmul 가속  
     -  I-BERT 정수 전용 비선형 근사 회로 추가 → 39.6× 전체 추론 속도 향상  
   - accelerator-aware quantization & co-design

3) **매핑·스케줄링**  
   - 매핑 공간(mapspace) 분석: BERT matmul vs ResNet convolution의 EDP 분포 유사  
   - fusion-optimized scheduling:  
     -  MHA query×key + Softmax 융합 시 22–78% latency 감소  
     -  FFN W₂ projection + LayerNorm 융합은 오히려 27% 성능 악화

4) **NAS**  
   - BigNAS 기반 supernet 학습 + 진화 알고리즘으로 Pareto-optimal 모델 탐색  
   - latency, energy, perplexity 동시 최적화:  
     -  0.1ppl 악화 허용 시 2.2× EDP 절감  
     -  1ppl 악화 시 10.6× EDP 절감  

## 4. 모델의 일반화 성능 향상 가능성
- **NAS로 얻은 아키텍처**들은 다양한 레이어별 하이퍼파라미터(헤드 수, 차원, FFN 크기)를 달리하여 일반화 능력을 유지하면서도 하드웨어 성능을 개선  
- **I-BERT 정수 전용 근사**는 다른 Transformer 변형(GPT, Vision Transformer 등)에도 적용 가능하여, 비선형 연산 근사와 정수화가 general-purpose 최적화 수단으로 확장 가능  
- **fusion-optimized scheduling 제약 분석**은 다양한 모델 구조(예: 긴 시퀀스, 저자원 환경)에서 일반화된 스케줄링 트레이드오프 모델로 활용  

## 5. 향후 영향 및 고려 사항
- **풀스택 시뮬레이션 및 자동화**: hardware↔software co-design 워크플로우 자동화, 스케줄러-NAS 결합 연구 필요  
- **비선형 연산 융합 가이드라인**: LayerNorm/Softmax 융합 소비자 자원(버퍼 크기, 시퀀스 길이)에 따른 일반화 전략 개발  
- **모델 구조 일반화**: Transformer 변형(비언어 분야)에도 NAS 기반 co-optimization 프레임워크 적용 검토  
- **에너지-정밀도 균형**: 대규모 모델에서도 허용 가능한 퍼플렉시티 악화를 동반한 에너지 절감 전략 연구  

— 본 논문은 Transformer 추론의 전 범위를 조망하며, 앞으로의 연구에서 하드웨어 가속, 스케줄러, NAS를 통합한 end-to-end 최적화 기법의 기반을 마련한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/08a6784b-dd7b-453c-a4e2-ef46449033cf/2302.14017v1.pdf

## 2 TRANSFORMER MODEL ARCHITECTURE AND PERFORMANCE BOTTLENECKS

Transformer 모델의 인퍼런스 성능 병목 현상을 이해하기 위해, 본 절은 크게 두 부분으로 나뉩니다.  
1) 모델의 구성요소별 상세 구조  
2) 각 구성요소가 하드웨어 성능에 미치는 영향 및 병목 분석  

***

### 2.1 Transformer 모델의 고수준 구조  
Transformer는 인코더·디코더 블록을 기반으로, 각 블록이 아래 세 가지 주요 연산으로 이루어집니다:  
  -  **Multi-Head Attention (MHA)**  
  -  **Feed-Forward Network (FFN)**  
  -  **Layer Normalization + 잔차 연결 (Residual Connection)**  

각 블록의 흐름은 다음과 같습니다(그림 1 참조).  

1) 입력 토큰 시퀀스(차원 ℓ × d)를  
2) MHA 모듈에 투입하여  
   a) Query, Key, Value로 투영(𝑑×𝑑×ℓ 매트릭스 곱, 총 3회)  
   b) ℎ개 헤드별로 분할 후, Key×Query(ℓ×(𝑑/ℎ)×ℓ) → Softmax → AttentionScore×Value((𝑑/ℎ)×ℓ×ℓ)  
   c) ℎ개 결과를 이어붙여 프로젝트(𝑑×𝑑×ℓ)  
3) LayerNorm → 잔차 연결  
4) FFN 모듈에 투입하여  
   a) 1차 투영(𝑑FFN×𝑑×ℓ) → GELU → 2차 투영(𝑑×𝑑FFN×ℓ)  
5) LayerNorm → 잔차 연결  

– MHA: 투영(matmul) 4회 + act-to-act matmul 2회 + Softmax  
– FFN: 투영(matmul) 2회 + GELU  

### 2.1.1 비선형 연산의 특성  
Softmax, LayerNorm, GELU 등의 비선형 연산은 모두 입력 전체에 대한 통계(평균·분산·지수 함수 합)를 런타임에 계산해야 하므로,  
- **반드시 여러 번 입력 값을 스캔**해야 하고  
- **일시적 메모리**(scratchpad)에 전체 활성화를 보관해야 하며  
- 일반 matmul보다 **낮은 연산-메모리 비율**(arithmetic intensity)을 갖습니다(그림 2).  

이로 인해, 비선형 연산은 전체 FLOPs 비중은 작아도(1~2% 미만)  
메모리 접근·대기 시간이 늘어나면서 전체 레이턴시에 큰 영향을 줄 수 있습니다.  
또한, CNN의 BatchNorm처럼 간단히 matmul에 흡수(fusion)하기도 어렵습니다.

### 2.1.2 인코더·디코더 변형  
Transformer는 원형(원조)으로 인코더-디코더 구조를 제안하였으나, 이후 다음 두 가지 변형이 널리 활용됩니다(그림 3).  
- **인코더 전용**: 전체 시퀀스를 병렬 처리, 자연어 이해(NLU)  
- **디코더 전용**: 토큰을 순차 생성(auto-regressive), 자연어 생성(NLG)  

인코더는 시퀀스 길이 ℓ에 대해 투영 matmul이 O(ℓ)이고 act-to-act matmul이 O(ℓ²)로 동작하며,  
디코더는 토큰 생성마다 투영이 매트릭스-벡터로 O(1), act-to-act는 과거 키·밸류 캐싱으로 O(ℓ) 비용이 듭니다.

***

### 2.2 모델 분석 및 병목

#### 2.2.1 플롭스·메모리 연산량·연산-메모리 비율(Arithmetic Intensity)  
- **FLOPs**: 시퀀스 길이 증가 시 act-to-act matmul (ℓ²)과 Softmax(ℓ²)로 인해 슈퍼선형 증가  
- **메모리 연산량(MOPs)**: 8-bit 가정 시도, 활성화·파라미터 로드/스토어 모두 집계 → FLOPs 비슷해도 메모리 비중 커짐  
- **Arithmetic Intensity** = FLOPs ÷ MOPs  
  -  인코더-only(BERT-Base): ℓ ≤ 512까지 증가 → ℓ > 512 이후 MHA act-to-act가 지배 → 감소(ℓ² 비중↑)  
  -  디코더-only(GPT-2): 전반적으로 낮음 → 메모리 대역폭 bound  
  -  CNN(ResNet-50) 대비: 비선형 연산 fusion으로 메모리 접근 ↓ → intensity ↑  

#### 2.2.2 직접 프로파일링(예: CPU)  
Intel Gold 6242 CPU에서 프로파일링 시:  
- ℓ ≤ 512: FFN·MHA 투영이 지배적  
- ℓ ↑: act-to-act matmul·Softmax가 지배적  
- GPT-2 디코더는 매트릭스-벡터 연산 crisis 로 메모리 바운드  
Normalized latency vs. ℓ 그래프(그림 9)와 동일한 추세 관찰

***

**핵심 요약**  
- Transformer 인퍼런스는 act-to-act matmul과 Softmax 등 ℓ² 비용 연산이 병목  
- 비선형 연산은 FLOPs 작아도 메모리 접근 많아 전체 레이턴시에 큰 영향을 줌  
- 인코더-only와 디코더-only 간 연산-메모리 비율 차이로 CPU/GPU 특성 따라 bound 유형이 다름  
- CNN accelerator 최적화 기법(Nonlinear fusion 등) 그대로 적용하기 어렵고, Transformer 특화 대응 필요

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/08a6784b-dd7b-453c-a4e2-ef46449033cf/2302.14017v1.pdf

# 3. HARDWARE DESIGN

**핵심 요약**  
Transformer 모델 추론을 효율적으로 수행하기 위해서는 전체 소프트웨어 스택뿐만 아니라 하드웨어 설계 관점에서도 여러 고려 사항이 필요합니다. 이 절에서는  
1) 범용 CPU/GPU 대비 DNN 전용 가속기의 구조  
2) Transformer 가속을 위한 특수 설계 기법  
3) 분석 모델을 통한 성능 예측  
4) Gemmini 기반 사례 연구  
등을 단계별로 설명합니다.

***

## 3.1 전통적 DNN 가속기의 구조  

  - **메모리 계층**  
    -  오프칩 DRAM: 전체 네트워크 가중치·활성화 저장  
    -  온칩 글로벌 버퍼(scratchpad): PE(Processing Element)로 데이터 공급  
    -  PE 로컬 레지스터(RF): 가장 저비용·저지연 저장  
    -  NoC(Network-on-Chip): PE 간 데이터 교환

  - **데이터플로우**  
    1. Temporal Dataflow  
       – SIMD/SIMT 방식으로 PE가 중앙 제어하에 메모리↔연산 반복  
       – 주로 서버 CPU/GPU에서 사용  
    2. Spatial Dataflow  
       – PE 간 직접 연결로 로컬 재사용 가속  
       – FPGA/ASIC 가속기에서 주로 채택  
       – Weight-stationary, Output-stationary, Row-stationary 등 다양한 재사용 전략 존재  

> “메모리 접근이 MAC(곱셈–덧셈)보다 수십~수백배 에너지 비싸므로 온칩 재사용이 필수”  

***

## 3.2 Transformer 전용 가속기 설계 고려사항  

  1. **메모리 및 대역폭 요구**  
     – Transformer는 작은 행렬(Attention)과 큰 FFN 매트멀을 번갈아 수행  
     – CNN 대비 다른 최적 타일 크기·계층별 메모리 요구  
  2. **비선형 연산 지원**  
     – Softmax, LayerNorm, GELU는 런타임 통계(평균·분산) 계산 필요  
     – 전용 SFU(Special Function Unit) 추가 시 면적 증가  
     – 없으면 CPU 오프로딩→추가 메모리 오버헤드  
  3. **데이터경로(Data Path) 설계**  
     – MHA 전용 가속기: 연산 그래프 하드코딩, Softmax 중간 삽입 가능  
     – 범용 엔드-투-엔드 가속기: Gemmini 스타일로 매트멀이 주 데이터경로  
     – 그래프 레벨 퓨전 여부, 비선형 연산 배치 전략이 성능·유연성 trade-off  

> “CNN 전용 가속기에서 잘 쓰이던 BatchNorm-퓨전은 Transformer LayerNorm에는 제약이 많아 오히려 성능 저하될 수 있음”  

***

## 3.3 분석 모델(Analytical Modeling)  

  - **목적**: 실제 하드웨어가 없거나 설계 초기 단계에서도 런타임·병목 예측  
  - **모델 구성**:  
    -  PE 배열(𝑊×𝑊 systolic array)  
    -  온칩 버퍼(Global SRAM, Accumulator SRAM)  
    -  외부 DRAM ↔ 온칩 대역폭, NoC 대역폭  
    -  연산과 메모리 전송은 완벽 중첩 가능 가정  

  - **활용 예시**  
    1. 지연시간(연산 vs. 메모리) 분해  
    2. 비이상(Non-ideal) 산술 강도 계산(타일링·32bit 출력 등 고려)  
    3. CNN vs. Transformer 비교  

  - **주요 인사이트**  
    – 대형 시퀀스에서 Attention 비선형 연산·Act-to-Act 매트멀의 산술 강도가 낮아 예측치 대비 더 큰 메모리 병목  
    – 이상적 산술 강도 대비 최대 2.5× 악화  

***

## 3.4 Gemmini 사례 연구: CNN → Transformer 가속기  

### 3.4.1 초기 Gemmini(ResNet-50용) 가속기  
  - 16×16 systolic 배열, 256 kB 8 bit 스크래치패드, 64 kB 32 bit 어큐뮬레이터  
  - CNN용 ReLU·Max-Pool 전용 유닛, int8-fp32 스케일링 지원  
  - Transformer 비선형 연산(Softmax·LayerNorm·GELU) 미지원 ⇒ CPU 오프로딩

### 3.4.2 초기 성능 병목  
  - 매트멀 유닛 74% 활용에도 전체 유닛 가동률 &lt;1%  
  - 비선형 연산 96% 시간 소모 (CPU 지연+데/양자화)  

### 3.4.3 메모리 계층 재조정  
  - Transformer Act-to-Act 매트멀은 큰 출력 행렬 ⇒ 어큐뮬레이터 크기↑가 중요  
  - 스크래치패드 256→64 kB, 어큐뮬레이터 64→256 kB  
  - 매트멀 레이턴시 36% 감소  

### 3.4.4 하드웨어-소프트웨어 공설계  
  - I-BERT 정수 전용 변형으로 GELU·Softmax·LayerNorm을 정수 다항 근사 구현  
  - 전용 정규화 모듈(누적→리듀스→다항근사) 추가: 면적 14%↑, 전력 9%↑  
  - 전체 Transformer 추론 레이턴시 39.6× 감소  

> “비선형 연산은 FLOPs 기여는 작지만, 오프로딩 시 지연폭발. 근사+전용유닛으로 오프칩 비용 제거해야 전체 성능 극대화”  

***

**결론**  
Transformer 추론 가속을 위해서는 CNN 가속기 설계 관행을 그대로 답습할 수 없습니다.  
– 메모리 계층 재조정(읽기/쓰기 버퍼 비율)  
– 비선형 연산 전용 유닛 또는 근사 구현  
– 하드웨어-소프트웨어 협력으로 오프로딩 비용 제거  
등을 종합하는 **풀스택 관점의 공설계**가 반드시 필요합니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/08a6784b-dd7b-453c-a4e2-ef46449033cf/2302.14017v1.pdf

# 4. MODEL OPTIMIZATION

**핵심 요약**  
이미 학습된 Transformer 모델을 하드웨어에 더 효율적으로 배포하기 위해서는, 네트워크 구조(아키텍처) 자체를 변경하지 않고도 연산량·메모리·전력·지연을 줄이는 알고리즘적 최적화가 필수적입니다.  
대표 기법으로는  
1) 저정밀화(Quantization)  
2) 희소화(Sparsity, Pruning)  
3) Transformer 고유 구조 최적화(Attention·Nonlinear·Decoding)  
가 있으며, 이 절에서는 각 기법을 단계별로 자세히 설명합니다.

***

## 4.1 Quantization (저정밀화)

  – **목적**: 32-bit 부동소수점(𝑭𝑷𝟑𝟐) 대신 8-bit 고정소수점(INT8) 등으로 표현하여  
    -  모델 크기·메모리 대역폭 4× 축소  
    -  ALU 연산 소요 에너지·칩 면적 30–×100 낮춤  
    -  메모리 계층 설계 재조정(더 많은 파라미터 온칩 저장)  

  – **방식**  
    1) 균등(Uniform) Quant: 일정 간격으로 구간 분할  
       -  간단·하드웨어 매핑 용이  
       -  양자화 오차: outlier 민감  
    2) 비균등(Non-uniform) Quant: 중요 구간에 더 많은 bin 할당  
       -  높은 정확도·복잡도↑  
    3) Mixed-precision: 레이어별 민감도 따라 비트 폭 가변  
       -  Hessian 기반 Q-BERT, RL 기반 HAT 등

  – **Integer-only Inference**  
    일부 임베디드 프로세서(FPUs 없음)용: 모든 연산을 정수로만 수행  
    -  I-BERT: GELU·Softmax·LayerNorm을 2차 다항식으로 근사 → 정수 곱셈·덧셈만 사용  
    -  결과: Gemmini에서 정수 전용 시 39.6× 레이턴시↓  

  – **주의점**  
    -  너무 낮은 비트(4-bit 이하)는 정확도 급락  
    -  Transformer는 activation outlier 존재 → FP8 등 새로운 부동소수점 정밀도 등장  

***

## 4.2 Sparsity (희소화·프루닝)

  – **목적**: 네트워크 파라미터 중 중요도 낮은 값(또는 토큰)을 제거하여 연산량·메모리 대역폭 감소  

  – **방식**  
    1) Weight Pruning  
       -  Unstructured Pruning: 임의 파라미터 제거 → 정확도 유지 높음  
         – 특수 하드웨어·압축 포맷 필요  
       -  Structured Pruning: 필터·채널·헤드·레이어 단위 제거  
         – 일반 매트멀 축소로 손쉬운 가속  
         – 최대 압축율은 낮음(∼70% pruning)  
    2) Activation Pruning (Dynamic)  
       -  Token Pruning: 중요도 낮은 토큰 런타임 삭제 → 연산 30–50% 절감  
       -  하드웨어: on-the-fly 스파스 감지·스케줄링 필요  

  – **중요 기법**  
    -  Magnitude Pruning: 절댓값 작은 weight 제거  
    -  Movement Pruning: 파인튜닝 시 weight 이동량 기준  
    -  First/Second-order: Gradient/Hessian 기반 중요도 측정  

  – **하드웨어 가속 기법**  
    -  Compressed Sparse Formats (CSC, CSR) + Sparse PE Skip  
    -  Load Balancing: PE 간 불균형 해소  

***

## 4.3 Transformer-specific Optimization

### 4.3.1 Attention 가속  
  – **Quadratic Cost**: 시퀀스 길이 𝑙 → 𝑙×𝑙 Act-to-Act 매트멀 비용(𝑂(𝑙²)) 급증  
  – **Token Pruning**: 중요 token만 Top-𝑘 유지를 위한 하드웨어 엔진  
  – **Sparse Attention**: attention scores 동적 0 근사 → 불필요 연산 스킵  
    -  각도 근사(ELSA), LSH-based 클러스터링, bit-serial 조기종료(LeOPARD) 등  

### 4.3.2 비선형 연산 가속  
  – Softmax·LayerNorm·GELU는 multi-pass→온칩 구현 어려움  
  – **근사**: Softmax base-2 Softermax, Taylor 5차, 정수 다항 I-BERT  
  – **LUT**: 소형 lookup table, NN-LUT 등으로 면적·지연 절감  

### 4.3.3 Decoder 가속  
  – Auto-regressive → 토큰당 𝑂(𝑙) Act-to-Act 매트멀 → 메모리 바운드  
  – **Early Exit**: 생성 깊이(디코더 레이어) 동적 조정  
  – **Big-Little Decoder**: 대·소 모델 협업 → 단순 토큰은 작은 모델 처리, 복잡 토큰만 큰 모델 콜  

### 4.3.4 최적화 기법 선택 가이드  
  – 단일 datapath vs. 모듈별 datapath 여부에 따라 적용 가능 기법 차이  
    -  통합 datapath: Static pruning·quantization 등 범용 기법  
    -  분리(reconfigurable) datapath: Dynamic sparsity·block-wise fusions 등  

***

**정리**  
Model Optimization은 기존 학습된 Transformer를 변경 없이 “경량화”하여 추론 효율을 극대화하는 핵심 단계입니다.  
– Quantization: 메모리·연산 에너지 4×–100× 절감, integer-only 가능  
– Sparsity: 파라미터·활성화 불필요 제거로 대역폭·연산량↓  
– Transformer-specific: Attention·Softmax·LayerNorm·Decoder 특화 기법으로 추가 가속  

이 기법들은 단독 적용보다는 하드웨어 설계·매핑 단계와 공설계하면, 추론 속도와 에너지 효율을 **수십배**까지 끌어올릴 수 있습니다.

# 5. MAPPING TRANSFORMERS TO HARDWARE

Transformer 추론을 하드웨어에 효율적으로 올리기 위해서는, 수학적으로 정의된 연산(매트멀, Softmax, LayerNorm 등)을 물리적 장치(PE 어레이, 메모리 계층, NoC 등)의 명령어 시퀀스로 변환하는 “매핑(mapping)” 과정이 핵심입니다. 이 장에서는

  5.1 매핑이란 무엇인가  
  5.2 매핑 시 내려야 할 주요 결정  
  5.3 효과적인 매핑 탐색 기법  
  5.4 매핑 성능 모델링  
  5.5 CNN과의 비교: Transformer 매핑의 난이도

를 단계별로 쉽고 자세하게 설명합니다.

***

## 5.1 매핑(mapping)이란 무엇인가

- **매핑(mapping)**: Transformer 네트워크(Attention, FFN, Softmax, LayerNorm 등)에서 정의된 연산을  
     -  메모리 이동(load/store)  
     -  파이프라인된 매트멀(mac units) 호출  
     -  특수연산(Softmax, LayerNorm) 수행  
    로 이루어진 하드웨어 명령어 시퀀스로 변환  
- 매핑의 유효성: 동일한 수학적 결과를 보장하면서 하드웨어 자원을 최대 활용  
- 거대한 “맵스페이스(mapspace)”:  
     -  버스트 단위(tile) 크기  
     -  루프 순서(loop permutation)  
     -  병렬화(Spatial vs Temporal)  
     -  그래프 수준(operator fusion, dynamic sparsity)  
    가능한 조합은 수십억~수조 개에 달함

***

## 5.2 매핑 시 내려야 할 주요 결정

### 5.2.1 그래프 수준 결정
    -  Operation Fusion  
      – 분리된 매트멀→Softmax→매트멀을 하나의 커널로 융합  
      – 메모리 왕복(off-chip↔on-chip) 감소  
    -  Dynamic Sparsity Pruning  
      – Attention 점수 대역폭이 작은 토큰 사전 제거  
    -  Static Sparsity Pruning  
      – 사전 학습된 중요치 기반 가중치·헤드·채널 제거  

### 5.2.2 연산 수준 결정
    -  Tiling(타일링):  
      – 대형 매트멀을 on-chip scratchpad fit 크기로 분할  
      – 각 축(Sequence, Hidden, FFN)별 tile_size 결정  
    -  Loop Permutation:  
      – 어느 축을 inner loop로 둘지→데이터 재사용·메모리 스트라이드 영향  
    -  Spatio-Temporal Mapping:  
      – PE 어레이상의 parallel 축 선택 vs. CPU처럼 temporal 실행  
    -  Double Buffering 등 통신·연산 오버랩

***

## 5.3 효과적인 매핑 탐색 기법

### 5.3.1 탐색 전략 분류
    -  완전탐색/무작위 (Brute-force/Random)  
      – Timeloop, dMazeRunner: 개발자 휴리스틱으로 맵스페이스 축소  
    -  피드백 기반 (Feedback-driven)  
      – AutoTVM(XGBoost), Ansor(Beam Search), Halide(RL/Beam)  
      – 샘플링→실측 성능→모델 학습 반복  
    -  제약 최적화 (Constrained Optimization)  
      – Polyhedral(Pluto), MIP, IOOpt: 수학적 제약식으로 타일·순서 결정  

### 5.3.2 그래프·연산 이음새
    - Tensor Algebra 중심 컴파일러(TVM, Halide, Exo)  
      → 매핑 명세(tiles, loops, vectorize)를 일련의 리라이트 규칙으로 전환  
      → 자동 코드 생성  

***

## 5.4 매핑 성능 모델링

실제 하드웨어 실행 없이 매핑 성능을 추정하기 위해

    -  분석적 모델(Analytical)  
      – PE배열·메모리 대역폭을 매개변수로 플롭·MOPs 기반 예측  
    -  폴리노미얼 모델(Polynomial)  
      – 텐서 차원 조합에 따라 닫힌 수식 제공→제약 최적화 목표로 사용  
    -  ML 기반 모델(Meta-Model)  
      – 샘플된 매핑 성능 데이터로 회귀/트리 모델 학습  

오버헤드가 큰 RTL 시뮬레이션 없이 반복 탐색 가능

***

## 5.5 CNN과의 비교: Transformer 매핑 난이도

### 5.5.1 맵스페이스 규모 비교  
    -  CNN convolution(6중 loop) vs Transformer matmul(3중 loop)  
    -  100K 무작위 매핑 실험 → EDP 분포 유사  
    -  최적역(상위 10%) 매핑 비율도 유사 → Transformer 매핑 난이도는 CNN과 동급  

### 5.5.2 LayerNorm·Softmax 융합의 함정  
    -  CNN: conv+ReLU/BatchNorm 융합은 순수 이득  
    -  Transformer: 매트멀→LayerNorm/Softmax 융합 시  
      – Accum SRAM 크기, Sequence 길이에 민감  
      – 불균형 타일링 강제→arithmetic intensity↓ 오히려 지연 증가  
    -  실험 결과(512 토큰 기준):  
      – Q×K + Softmax 융합 → 78% Softmax 지연 숨김  
      – Wout + LayerNorm 융합 → 타일 제약으로 매트멀 지연↑  
    -  최종 판단: Softmax 융합만 취하고, LayerNorm은 별도 스케줄 권장

***

**정리**  
Transformer 매핑은 단순 matmul만 고려하는 CNN 매핑보다 더 복잡합니다.  
    -  거대한 타일·병렬화·루프 순서 조합으로 맵스페이스 폭발  
    -  MHA 특유의 Softmax·LayerNorm 융합 시 tile 제약→성능 역효과  
적절한 탐색·모델링·하드웨어 제약 고려를 통해서만 효율적 매핑을 얻을 수 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/08a6784b-dd7b-453c-a4e2-ef46449033cf/2302.14017v1.pdf

# 6 Adapting Transformer Architecture with Neural Architecture Search (NAS)

이 장에서는 **Transformer 모델 구조를 하드웨어 특성에 맞추어 자동으로 최적화**하는 방법으로 NAS(신경망 구조 탐색)를 소개합니다. 전통적으로 네트워크 구조는 사람이 실험과 직관을 통해 설계해 왔지만, NAS를 통해 자동화된 탐색으로 정확도와 하드웨어 효율(추론 지연·에너지)를 동시에 고려할 수 있습니다.  

## 6.1 Neural Architecture Search  
NAS는 크게 세 가지 구성 요소로 이루어집니다.  
1. **탐색 공간(Search Space)**  
   - 설계 가능한 연산(예: Conv3×3, Self-Attention, Activation 등)과 이들의 연결 규칙으로 정의  
   - 전통적 CNN에서는 ‘레이어 단위’ 또는 ‘셀(cell) 단위’로 나뉘어 설계  
   - 셀 단위 탐색(Cell-wise Search): 기본 블록(셀)을 반복 쌓아 전체 네트워크 구성 → 탐색 공간 대폭 축소  
2. **탐색 방법(Search Method)**  
   - **강화학습(RL-based)**: 에이전트가 구조를 샘플링하고 그 성능(정확도)을 보상(reward)으로 학습  
   - **진화 알고리즘(Evolutionary)**: 초기 모델군(population)을 변형(돌연변이)·선택하며 세대별로 진화  
   - **연속 완화(Gradient-based, e.g. DARTS)**: 구조 선택을 “가중치화된 연산 조합”으로 연속 공간에 풀고, 경사하강법으로 최적화  
3. **평가 방법(Evaluation Method)**  
   - 전통적 방법: 각 후보 구조를 완전 학습→검증→선별(비싼 비용)  
   - **슈퍼넷(Supernet) + 가중치 공유(Weight Sharing)**: 하나의 과대모델(supernet)에 모든 후보를 중첩, 서브넷(subnet)별로 분리해 학습 없이 평가  
   - **프록시 과제(Proxy Task)**: 소규모 데이터셋(CIFAR-10)에서 탐색 후 대용량 과제(ImageNet)로 전이  

**탐색 공간**은 cell-wise 설계로 줄이고, **탐색 방법**은 연속 완화나 슈퍼넷+진화 기법으로 학습 비용을 절감하며, **평가**는 가중치 공유된 슈퍼넷에서 속도와 정확도를 동시에 측정하는 방식을 통해 NAS 효율을 높입니다.

## 6.2 Hardware-Aware NAS  
NAS 결과가 실제 하드웨어에서 빠르고 에너지 효율적이려면 **하드웨어 성능 지표(추론 지연, 에너지)**를 최적화 목표에 포함해야 합니다.  
- **직접 측정**: 실제 디바이스에서 레이턴시나 에너지 측정 → NAS 보상(reward) 또는 정규화 항으로 사용  
- **연산 단위 룩업테이블**: 각 연산(Conv, MatMul 등)에 대한 측정치를 미리 저장, 네트워크 전체 예측값 합산  
- **성능 예측 모델**: 연산 속성(입력·출력 크기, 연산 수 등)을 입력으로 하드웨어 성능 지연을 추정하는 경량 회귀 모델  

이로써 실제 장치에서 직접 측정하기 어려운 대규모 NAS 탐색도 병렬화하고 빠르게 수행할 수 있습니다.

## 6.3 Transformer-Specific NAS  
Transformer 구조에 특화된 NAS 기법들이 최근 활발히 연구되었습니다. 이들은 대부분 **슈퍼넷 + 진화 알고리즘** 기반으로, NLP와 CV용 ViT 모두에 적용됩니다.

- **Evolved Transformer** (RL→진화): 인코더·디코더용 “셀” 반복 구조를 진화 알고리즘으로 탐색  
- **HAT, NAS-BERT**: Once-For-All, SPOS 기반 슈퍼넷에 깊이, 헤드 수, 차원 등의 하이퍼파라미터 탐색  
- **Primer**: TF 연산 단위로 구성된 프로그램을 직접 진화 탐색  
- **Autoformer, ViT-ResNAS**: ViT 셀 단위 구조를 진화 알고리즘으로 탐색  
- **GLiT, NAS-ViT, BurgerFormer**: 계층적·미시구조까지 포함한 하이브리드 CNN+Attention 구조를 탐색  

대부분의 방법이 대규모 학습 비용 문제로 **가중치 공유(supernet)** 방식을 택하며, 구조 탐색→샘플링된 서브넷→재학습 단계를 최소화하는 데 주안점을 둡니다.

## 6.4 Case Study: NAS-Co-Design for Transformers on Gemmini  
### 설정  
- **Baseline**: 6-Layer Transformer (BERT-Base 유사), WikiText-2 언어 모델링  
- **슈퍼넷 학습**: BigNAS 방식(샌드위치 규칙)  
- **서브넷 탐색**: 진화 알고리즘, 𝑙(레이어), ℎ(헤드 수), 𝑑(모델 차원), 𝑑FFN 범위  
- **하드웨어 추정**: Gemmini NPU, Timeloop 기반 룩업 테이블로 지연·에너지 예측  

### 결과  
- **지연 vs. 퍼플렉서티**: 0.1 포인트 손실 허용 시 1.4× 속도 개선, 1.0 포인트 손실 시 2.4× 개선  
- **에너지 vs. 퍼플렉서티**: 0.1 포인트 손실 시 1.6× 절감, 1.0 포인트 손실 시 4.4× 절감  
- **EDP vs. 퍼플렉서티**: 0.1 포인트 손실 시 2.2× 개선, 1.0 포인트 손실 시 10.6× 개선  

이는 NAS를 통해 모델 구조와 하드웨어 매핑을 **동시에 최적화(co-design)** 할 때 얻을 수 있는 실제 성능 향상을 보여 줍니다.  

―――――――――――――――――――――――――  
**결론**  
Transformer 구조를 정밀하게 NAS로 탐색하면, **정확도 손실을 거의 없이** 모델 구조(레이어 수·차원·헤드)와 하드웨어 특성(지연·에너지)을 동시에 최적화할 수 있습니다. 특히 슈퍼넷+진화 기법과 하드웨어 예측 모델을 결합하면 NAS 비용을 크게 낮추면서도 혁신적인 스루풋·에너지 효율 개선을 달성할 수 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/08a6784b-dd7b-453c-a4e2-ef46449033cf/2302.14017v1.pdf
