# Q-Diffusion: Quantizing Diffusion Models

**핵심 주장**  
Q-Diffusion은 확산 모델의 반복적 노이즈 예측 네트워크를 **데이터 프리(post-training) 방식으로 4비트까지 양자화**하면서도, 전체 정밀도 모델 대비 Perceptual 품질 저하(FID 변화)가 **최대 2.34** 수준에 불과하도록 설계된 기법이다.[1]

**주요 기여**  
1. **Time Step–Aware Calibration**: 각 디노이징 단계에서 변화하는 활성 분포를 반영해, 전체 타임스텝에 고르게 걸쳐 캘리브레이션 샘플을 추출하는 방식 제안.[1]
2. **Split-Shortcut Quantization**: UNet 내 스킵 연결에서 서로 다른 통계 분포를 갖는 깊은·얕은 채널을 분할해 별도 양자화함으로써 과도한 양자화 오류를 방지.[1]
3. **Training-Free 4-bit PTQ**: 기존 PTQ가 8-bit에 그쳤던 한계를 돌파해 **W4A8**에서도 안정적 품질 유지(FID 변화 ≤2.34) 달성.[1]
4. **Text-Guided 확산 모델 적용**: Stable Diffusion에 4-bit 양자화를 적용한 최초 사례로, **텍스트 조건부 생성에서도 정밀도 저하 없이** 실행 가능함을 시연.[1]

***

## 1. 해결하고자 하는 문제

- **느린 추론 및 높은 메모리·계산 비용**: 확산 모델은 수십~수백 타임스텝 반복 수행하며 복잡한 UNet 기반 노이즈 추정기 계산이 병목.[1]
- **양자화 적용의 어려움**:  
  1. *Accumulated Quantization Error*: 각 단계 양자화 오류가 누적되어 후반 스텝에서 MSE가 기하급수적으로 증가.[1]
  2. *Varying Activation Distribution*: 디노이징 단계마다 활성 분포가 크게 달라, 단일 스텝 캘리브레이션만으로는 전 스텝 일반화 불가.[1]
  3. *Shortcut Layer Bimodality*: UNet 스킵 연결 시 채널별 활성 및 가중치 분포가 이중봉우리를 이루어, 통상적 단일 클리핑 범위 적용 시 심각한 왜곡 발생.[1]

***

## 2. 제안하는 방법

### 2.1 Time Step–Aware Calibration  
- 캘리브레이션 단계 간격 $$c$$마다 전체 타임스텝 $$1 \ldots T$$에서 균일 추출해 $$N$$개의 중간 입력 $$\{x_t^{(i)}\}$$ 수집 (Alg.1).[1]
- 수집된 데이터로 UNet의 블록 단위 재구성(reconstruction) 최적화:

$$
    \min_{s}\ \mathbb{E}\_{x\sim D}\big\|Q_{s}(W)f_s(x)-Wf(x)\big\|^2
  $$
  
  여기서 $$f_s$$는 블록 출력, $$s$$는 클리핑·스케일 파라미터.[1]

### 2.2 Split-Shortcut Quantization  
- 스킵 연결 입력 $$X=[X_1\oplus X_2]$$, 가중치 $$W=[W_1\oplus W_2]$$를 채널 축으로 분할 양자화:

$$
    Q(X)=Q(X_1)\oplus Q(X_2),\quad Q(W)=Q(W_1)\oplus Q(W_2)
  $$

- 분할 전·후 연산 오류를 최소화하도록 adaptive rounding 적용.[1]

***

## 3. 모델 구조 및 성능 개선

| 모델           | Precision (W/A) | CIFAR-10 FID↓ | LSUN-Bedroom FID↓ | LSUN-Church FID↓ |
|----------------|-----------------|---------------|-------------------|------------------|
| Full Precision | 32/32           | 4.22          | 2.98              | 4.06             |
| Linear Quant   | 4/8             | 54.22         | 82.69             | 32.54            |
| Q-Diffusion    | 4/8             | **5.09**      | **4.86**          | **4.45**         |

- 4-bit 가중치, 8-bit 활성화(W4A8)에서도 FID 변화 ≤2.34 유지.[1]
- Stable Diffusion 텍스트 조건부 생성에서도 W4A8 적용 후 **품질 저하 없이** 512×512 이미지 합성 가능.[1]

***

## 4. 한계 및 일반화 성능 향상 가능성

- **조건부 모델 완전 양자화**: 텍스트 인코더·크로스 어텐션 이후 SoftMax 활성의 양자화 설계 미비.  
- **샘플링 스케줄 변화**: 다른 ODE 해결자(DPM-Solver) 적용 시 활성 분포 불일치로 캘리브레이션 성능 저하 가능성 확인.[1]
- **비균일·데이터 의존 샘플링**: 현재 균일 추출만 사용, 분포 기반·적응적 샘플링 연구 여지.

***

## 5. 향후 연구 영향 및 고려 사항

- **경량화된 실시간 서비스**: 모바일·엣지 디바이스에서 복잡한 확산 모델 4-bit 양자화 가능성 열어, 대화형 이미지 생성·AR/VR 응용 확장.  
- **양자화-샘플링 통합 프레임워크**: Fast sampler와 양자화를 결합한 **End-to-End** 가속화 연구 필요.  
- **다중 모달 일반화**: 텍스트, 오디오, 비디오 등 다양한 조건부 확산 모델 전반에 걸친 **양자화 기법의 일반화 및 안정성 검증**.  
- **Adaptive Calibration**: 실행 중 활성 분포 변화 감지 기반 **온라인 캘리브레이션** 또는 **동적 양자화**로 일반화 성능 강화 모색.

 Q-Diffusion: Quantizing Diffusion Models (arXiv:2302.04304v3)[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/60d9e53e-d284-4f76-9e29-59c6e2371496/2302.04304v3.pdf)
