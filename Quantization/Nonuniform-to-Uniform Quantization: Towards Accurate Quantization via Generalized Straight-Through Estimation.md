# Nonuniform-to-Uniform Quantization: Towards Accurate Quantization via Generalized Straight-Through Estimation

**핵심 주장**  
‘Nonuniform-to-Uniform Quantization (N2UQ)’는 입력 구간(threshold)만 비등간격(nonuniform)으로 학습하고 출력 레벨은 균등 간격(uniform)으로 고정함으로써, 기존 비등간격(nonuniform) 퀀타이제이션의 높은 표현력과 균등 간격(uniform) 퀀타이제이션의 하드웨어 효율성을 동시에 달성할 수 있다고 주장한다.

**주요 기여**  
1. 입력 임계값(threshold) 학습을 도입한 N2UQ 설계  
2. 비등간격 입력에 대해 효과적인 역전파를 가능케 하는 Generalized Straight-Through Estimator (G-STE) 제안  
3. 엔트로피 보존 가중치 정규화(entropy preserving weight regularization)로 양자화 정보 손실 최소화  
4. ImageNet 위 ResNet-18/34/50, MobileNet-V2에서 최대 1.7% Top-1 정확도 향상 달성  

***

## 1. 해결하고자 하는 문제  
- **표현력 vs. 효율성 트레이드오프**  
  -  비등간격(nonuniform) 퀀타이제이션은 입력 분포에 맞춰 레벨을 할당해 높은 정확도를 얻지만, 결과값이 실수이므로 하드웨어 가속(bitwise 연산)에 적합하지 않아 LUT(look-up table) 매핑 오버헤드가 발생  
  -  균등 간격(uniform) 퀀타이제이션은 비트 연산 가속에 최적화되지만, 입력 분포에 대한 적응력이 낮아 양자화 오차가 큼  

- **기존 STE 한계**  
  STE(straight-through estimator)는 계단 함수의 도함수가 거의 0인 문제를 완화하나, 임계값 학습에는 무력해 입력 임계값을 학습하지 못함  

***

## 2. 제안 방법

### 2.1. 비등간격 입력–균등 간격 출력 정량화 (N2UQ)  
출력 레벨은 0,1,…,2ⁿ–1로 균등(equidistant)하며, 경계값(T₁,…,T₂ⁿ⁻¹)은 학습 가능:

$$
x_q =
\begin{cases}
0 & x_r < T_1,\\
i & T_i \le x_r < T_{i+1},\ i=1,\dots,2^n-2,\\
2^n-1 & x_r \ge T_{2^n-1}.
\end{cases}
$$

### 2.2. Generalized Straight-Through Estimator (G-STE)  
- **아이디어**: 확률적 양자화(stochastic quantization)의 기댓값(expectation)을 역전파 근사(backward approximation)로 사용  
- **수식**: $$2^n$$개 구간(segment)을 각기 다른 임계값 $$d_i$$와 구간폭 $$a_i$$로 나누고,  

$$
  \frac{\partial x_q}{\partial x_r} =
  \begin{cases}
  \frac{\partial}{\partial x_r}\bigl(\tfrac{x_r - d_{i-1}}{a_i} + i-1\bigr)
    & d_{i-1} \le x_r < d_i,\ i=1,\dots,2^n-1,\\
  0 & \text{otherwise}.
  \end{cases}
  $$

- **특징**: 모든 간격이 같을 때(STE), 일반 STE와 일치하며, 비등간격에서는 구간별 기울기를 달리해 임계값을 학습 가능

### 2.3. 엔트로피 보존 가중치 정규화  
- **목표**: 양자화 후 가중치 분포의 정보 엔트로피 최대화  
- **방법**: 실수 가중치 $$W^r$$를  

$$
  W^{r'} = \frac{2^{n-1}}{2^n-1}\,\frac{|W^r|}{\|W^r\|_1}\,W^r
  $$
  
  로 재스케일링하여, 양자화 레벨별로 균등 분포 유도  

***

## 3. 모델 구조  
- Pre-Activation ResNet 구조(Residual block 내 비선형→Conv→BN 순)  
- 모든 Conv 및 FC layer에 N2UQ 적용(첫/마지막 layer 제외)  
- 학습률: 가중치의 1/10로 임계값 및 스케일 β₁,β₂ 업데이트  

***

## 4. 성능 향상  
| 네트워크     | 비트폭 (W/A) | 기존 최고 | N2UQ 개선폭 |
|-------------|--------------|-------------------|-------------|
| ResNet-50   | 2/2          | 75.1%            | **+0.7%**  |
| ResNet-50   | 3/3          | 76.3%            | **+1.2%**  |
| ResNet-18   | 2/2          | 68.9%            | **+0.5%**  |
| MobileNet-V2| 4/4          | 71.6%            | **+0.5%**  |

- 특히 2-bit ResNet-50 Top-1 75.8% → real-valued(77.0%) 대비 격차 1.2% → 0.6%로 대폭 감소  

***

## 5. 한계 및 일반화 성능  
- **한계**  
  -  추가 학습 파라미터(임계값, 스케일)로 메모리·학습 안정성 고려 필요  
  -  매우 낮은 비트(1-bit)나 비(非)대칭 분포에서 과적합 위험  

- **일반화 성능**  
  -  엔트로피 보존 정규화가 과적합 방지와 분포 적응에 기여  
  -  다양한 아키텍처(ResNet, MobileNet)에서 일관된 이득 → 타 모델에도 확장 가능  

***

## 6. 향후 연구 영향 및 고려사항  
- **영향**  
  -  하드웨어 가속 친화적이면서도 비등간격 표현력 확보한 새로운 퀀타이저 설계 방향 제시  
  -  STE 한계를 극복하는 G-STE 개념 확장 가능성  

- **고려사항**  
  -  적응적 비트폭 할당(mixed-precision)과의 결합  
  -  1-bit 양자화 및 비대칭 분포에 대한 G-STE 안정성 분석  
  -  ASIC/FPGA 등 실제 하드웨어에서의 오버헤드·전력 평가  

> 이 논문은 하드웨어 효율성과 표현력 간 타협점을 제시함으로써, 후속 연구의 퀀타이제이션 설계에 중요한 이정표가 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/cc673951-b4a0-42e1-bd8a-c76554735310/2111.14826v2.pdf)
