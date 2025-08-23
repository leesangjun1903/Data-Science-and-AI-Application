# I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference | Image classification

**핵심 주장 및 주요 기여**  
I-ViT는 Vision Transformer(ViT)의 전체 연산 그래프를 8-bit 정수 연산만으로 수행할 수 있도록 하는 최초의 정수 전용 양자화(Integer-only Quantization) 기법이다. 동차성(homogeneity) 조건에 기반한 기존의 다항식 근사 방식 대신, 모든 연산을 비트 시프트(bit-shifting)와 간단한 정수 연산으로 대체하는 Shiftmax, ShiftGELU, I-LayerNorm을 제안하여, 추가적인 부동소수점 연산 없이도 원본 FP 모델과 동등한 수준의 정확도를 유지하면서 3.7∼4.1× 속도 향상을 실현한다.

***

## 1. 해결하고자 하는 문제  
Vision Transformer는 뛰어난 성능에도 불구하고 대규모 모델 크기와 부동소수점 연산 의존성으로 인해 엣지 디바이스에서의 실시간 추론이 어려웠다.  
- **비선형 연산(Softmax, GELU, LayerNorm)의 부동소수점 의존성**: 기존 FasterTransformer는 Softmax·GELU·LayerNorm을 FP 연산으로 남겨 비효율적이며, 정수와 FP 간 데이터 전달 오버헤드가 크다.  
- **CNN용 다이아딕(dyadic) 파이프라인의 한계**: Dyadic arithmetic는 선형·ReLU 계열에만 적용 가능해 ViT의 비선형 연산을 처리할 수 없다.

***

## 2. 제안 방법 및 모델 구조  

### 2.1 전체 구조  
ViT 블록 한 단위는  

$$
\hat{X} = \text{MSA}(\text{LayerNorm}(X)) + X,\quad
Y = \text{MLP}(\text{LayerNorm}(\hat{X})) + \hat{X}
$$  

으로 구성되며, MSA 내부의 Softmax, MLP 내부의 GELU, 각 블록 앞뒤 LayerNorm이 비선형 연산에 해당한다.

### 2.2 선형 연산: Dyadic Arithmetic  
모든 MatMul·Dense 연산은 다음과 같이 INT8 입력 $$I_Q, I_K$$과 스케일 $$S_Q,S_K$$을 곱한 후, 정수 곱셈·비트 시프트만으로 출력 재양자화(Requantization)를 수행한다.  

$$
I_{\text{out}} = \bigl(b \cdot (I_Q \ast I_K^T)\bigr) \gg c,
\quad
\text{where}\;\; b,c\in\mathbb{Z}^+
$$

### 2.3 비선형 연산  
1) **Shiftmax (Softmax 정수 근사)**  
   -  입력 정수 $$I$$, 스케일 $$S$$에서  

$$
   I_\Delta = I - \max(I),\quad
   e^{S\,I_\Delta} \approx 2^{S\,I_\Delta \log_2 e}\;\xrightarrow[\text{비트근사}]{\text{(13),(15)}}\;\text{ShiftExp}(I_\Delta,S)
   $$  
   
   -  분자·분모 항 모두를 비트 시프트로 근사 후, INTDiv로 정수 나눗셈 수행  

$$
   I_{\text{out}} = \mathrm{IntDiv}(I_{\exp},\sum I_{\exp},k_{\mathrm{out}})
   $$

2) **ShiftGELU (GELU 정수 근사)**  
   -  $$1.702\,x$$를 비트 시프트로 근사하고, sigmoid 약식을 Shiftmax 기반으로 구현  

$$
   \mathrm{GELU}(x)\approx x\cdot\sigma(1.702x)
   $$

3) **I-LayerNorm (LayerNorm 정수 근사)**  
   -  분산 계산 후, 표준편차 $$\sqrt{\mathrm{Var}}$$를 다음 반복식을 비트 시프트로 계산  

$$
   I_{i+1} = (I_i + \lfloor \mathrm{Var}/I_i\rfloor)\gg1,\quad 10\text{회 반복}
   $$

***

## 3. 성능 향상 및 한계  

| 모델     | 정확도 (FP) | 정확도 (I-ViT) | 지연시간(ms) (FP) | 지연시간(ms) (I-ViT) | 속도향상 |
|---------|-------------|-----------------|-------------------|----------------------|----------|
| DeiT-S  | 79.85%      | 80.12% (+0.27)  | 11.5              | 2.97                 | ×3.87    |
| Swin-S  | 83.20%      | 83.01% (−0.19)  | 27.8              | 6.92                 | ×4.02    |

- **정확도**: FP 수준을 유지하거나 소폭 초과  
- **지연시간**: 3.72∼4.11× 단축 (RTX 2080Ti)  
- **한계**:  
  -  현재 GPU·TVM 지원이 최적화되지 않아 배치 크기 증가 시에도 선형 병렬화 한계  
  -  FPGAs 등 전용 정수 전용 하드웨어에서 추가 최적화 여지 존재  

***

## 4. 일반화 성능 향상 가능성  
- **도메인 확장**: Shiftmax·ShiftGELU의 도출 방식은 근사 오차가 전 영역에 걸쳐 제한적이므로, ViT 외에도 Transformer 구조가 쓰이는 언어·멀티모달 모델에 그대로 적용 가능  
- **추가 튜닝 여지**: 양자화 클리핑 범위(m)와 출력 비트 폭(kout)를 작업별 분산 특성에 맞춰 조정하면 일반화 성능을 더 끌어올릴 수 있음  
- **심층 구조 대응**: 반복적 계산 기반인 I-LayerNorm 수렴 속도를 모델 깊이에 따라 동적으로 제어하는 전략을 도입하면 더욱 견고한 일반화 달성 가능  

***

## 5. 향후 연구 영향 및 고려사항  
- **연구 영향**:  
  -  _정수 전용 추론_ 패러다임을 ViT에 최초로 도입함으로써, 엣지·임베디드 비전 모델의 경량화·가속화 연구에 새로운 방향 제시  
  -  비선형 연산 근사 기법(Shiftmax·ShiftGELU)이 다양한 신경망 구조에 응용될 수 있는 기반 마련  

- **고려사항**:  
  1. **하드웨어-소프트웨어 공동 설계**: 전용 정수 로직 최적화, 비트 시프트 유닛 확장 등을 반영한 칩 설계와 컴파일러 지원 필요  
  2. **다중 정밀도 혼합**: 특정 계층에 혼합 정밀도를 도입하여, 계산 비용과 성능 간 최적 균형을 모색  
  3. **다양한 비전 과제**: 객체 검출·분할·비디오 인식 등 복합 비전 작업에서 근사 및 일반화 성능 평가  

---  

I-ViT는 정수 전용 양자화 기법을 통해 ViT를 포함한 다양한 Transformer 기반 모델의 엣지 디바이스 적용 가능성을 크게 확장시키는 중요한 발판을 제공한다. 앞으로 전용 하드웨어 및 컴파일러 최적화를 결합하여, 실시간 딥러닝 추론의 새로운 표준으로 자리매김할 전망이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c3c5a186-a703-4888-864e-2ad5ae6b94b9/2207.01405v4.pdf)
