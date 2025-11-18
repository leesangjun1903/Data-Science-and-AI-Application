
# Implicit Neural Representations with Periodic Activation Functions

## 1. 논문의 핵심 주장과 주요 기여

**핵심 주장**: 주기함수(Periodic Activation Functions)를 사용한 신경망은 **암묵적 신경 표현(Implicit Neural Representations)**에서 ReLU 기반 네트워크보다 훨씬 우수한 성능을 제공한다. 특히 고주파 신호 성분과 도함수를 정확하게 표현할 수 있다.

**주요 기여**:
- **SIREN 아키텍처 제안**: 사인(sine) 함수를 활성화 함수로 사용하는 신경망 구조의 도입
- **원리에 기반한 초기화 방식**: 네트워크 깊이에 관계없이 활성화 분포를 유지하는 수학적 초기화 방식 제시
- **다양한 응용 시연**: 이미지, 비디오, 오디오, 3D 형상, 편미분방정식 해결 등 광범위한 적용
- **도함수 표현 능력**: 신호의 1차, 2차 도함수까지 정확하게 표현 가능[1]

***

## 2. 논문의 문제 정의, 제안 방법 및 모델 구조

### 2.1 해결하려는 문제

기존 ReLU 기반 암묵적 신경 표현의 한계:

$$F\left(x, \Phi, \nabla_x\Phi, \nabla^2_x\Phi, \ldots\right) = 0$$

$$\Phi : x \rightarrow \Phi(x)$$

여기서 $\Phi$는 암묵적으로 관계식 $F$에 의해 정의되는 함수이다. ReLU 네트워크는:
- 2차 도함수가 모든 곳에서 0이므로 고주파 정보 표현 불가
- 미분 가능하지만 도함수의 거동이 불안정함
- 세부 정보(fine detail) 표현 능력 부족[1]

### 2.2 제안하는 방법 및 수식

**기본 SIREN 구조**:

$$\Phi(x) = W_n(\varphi_{n-1} \circ \varphi_{n-2} \circ \ldots \circ \varphi_0)(x) + b_n$$

$$\varphi_i(x_i) = \sin(W_i x_i + b_i)$$

여기서 $\varphi_i : \mathbb{R}^{M_i} \rightarrow \mathbb{R}^{N_i}$는 $i$번째 계층이다.[1]

**핵심 특성**: SIREN의 도함수는 다시 SIREN이다.

$$\frac{d}{dx}\sin(Wx + b) = W \cos(Wx + b) = W\sin(Wx + b + \frac{\pi}{2})$$

코사인은 위상이 이동한 사인이므로, SIREN의 모든 도함수도 주기함수 활성화를 가진 신경망 형태를 유지한다.[1]

### 2.3 원리에 기반한 초기화 방식

**활성화 분포 분석**:

균일 분포 입력 $x \sim U(-1, 1)$에 대해, $y = \sin(\frac{\pi}{2}x)$의 출력은 **역사인 분포(Arcsine Distribution)**를 따른다:

$$y \sim \text{Arcsin}(-1, 1), \quad f_Y(y) = \frac{1}{\pi\sqrt{1-y^2}}$$

**가중치 초기화 전략**:

첫 번째 계층을 제외한 모든 계층에서:

$$w_i \sim U\left(-\sqrt{\frac{6}{n}}, \sqrt{\frac{6}{n}}\right)$$

여기서 $n$은 입력 차원(fan-in)이다. 이를 통해 각 사인 활성화 입력이 표준 정규 분포 $N(0,1)$를 따르도록 보장한다.[1]

첫 번째 계층에서는:

$$\sin(\omega_0 \cdot Wx + b), \quad \omega_0 = 30$$

$\omega_0$는 공간 주파수를 제어하여 첫 번째 계층이 입력 범위 $[-1, 1]$에서 여러 기간을 반복하도록 한다.[1]

**손실 함수**:

암묵적 신경 표현을 학습하기 위한 일반적인 형태:

$$L = \int_{\Omega} \sum_{m=1}^{M} \mathbb{1}_{\Omega_m}(x) \left\|C_m\left(a(x), \Phi(x), \nabla\Phi(x), \ldots\right)\right\| dx$$

실제 구현에서는 샘플링을 통해:

$$\tilde{L} = \sum_{i \in D} \sum_{m=1}^{M} \left\|C_m\left(a(x_i), \Phi(x_i), \nabla\Phi(x_i), \ldots\right)\right\|$$

### 2.4 모델 구조

**이미지 피팅의 예**:

- 입력: 좌표 $(x, y) \in \mathbb{R}^2$
- 출력: RGB 색상 $\Phi(x) \in \mathbb{R}^3$
- 제약 조건: $C(f(x_i), \Phi(x_i)) = \Phi(x_i) - f(x_i) = 0$[1]

**응용별 모델 구성**:

1. **Poisson 방정식 풀이**:
$$L_{\text{grad}} = \int_{\Omega} \|\nabla_x\Phi(x) - \nabla_x f(x)\| dx \text{ 또는 } L_{\text{lapl}} = \int_{\Omega} \|\Delta\Phi(x) - \Delta f(x)\| dx$$

2. **Signed Distance Functions (SDF)**:
$$L_{\text{sdf}} = \int_{\Omega} \big\||∇_x\Phi(x)| - 1\big\| dx + \int_{\Omega_0} \|\Phi(x)\| + \|1 - \langle\nabla_x\Phi(x), n(x)\rangle\| dx + \ldots$$

여기서 Eikonal 제약 $|∇Φ| = 1$이 적용된다.[1]

3. **Helmholtz 방정식**:
$$H^{(m)}\Phi(x) = -f(x), \quad H^{(m)} = \Delta + m(x)w^2$$

손실 함수: $L_{\text{Helmholtz}} = \int_{\Omega} \lambda(x)\|H^{(m)}\Phi(x) + f(x)\|_1 dx$[1]

***

## 3. 성능 향상 및 실험 결과

### 3.1 정량적 성능 비교

**이미지 표현**:
- SIREN: PSNR 약 29-30 dB 달성
- ReLU: PSNR 약 25 dB (약 4-5 dB 차이)[1]

**비디오 표현**:
- SIREN: 평균 PSNR 29.90 dB (표준편차 1.08)
- ReLU: 평균 PSNR 25.12 dB (표준편차 1.16)[1]

**도함수 표현 능력** (Poisson 이미지 재구성):

| 모델 | 기울기 감독 | 라플라시안 감독 |
|------|-----------|--------------|
| Tanh | 25.79 dB | 7.11 dB |
| ReLU P.E. | 26.35 dB | 11.14 dB |
| SIREN | **32.91 dB** | **14.95 dB** |

SIREN은 도함수 감독 하에서 다른 방법들보다 훨씬 우수한 성능을 보인다.[1]

### 3.2 주요 응용 결과

**1. Poisson 이미지 재구성 및 편집**:
- 이미지 기울기나 라플라시안만으로 원본 이미지를 정확하게 재구성
- 두 이미지의 기울기를 혼합하여 자연스러운 이미지 합성 가능[1]

**2. Signed Distance Functions (SDF)**:
- 점 구름(point cloud)에서 직접 SDF 학습
- 복잡한 장면을 단일 5계층 네트워크로 매개변수화
- 국소 격자 기반 방법과 달리 전역적 고주파 세부사항 보존[1]

**3. Helmholtz 및 파동 방정식 풀이**:
- 수치 격자 기반 솔버와 유사한 정확도 달성
- 다른 활성화 함수(ReLU, tanh, RBF)보다 훨씬 나은 성능[1]

**4. Full-Waveform Inversion (FWI)**:
- 희소한 센서 데이터로부터 파동장과 속도 모델 동시 재구성
- 기존 FWI 솔버(ADMM 기반)보다 우수한 속도 모델 복구[1]

### 3.3 일반화 성능: 하이퍼네트워크를 통한 학습

**Conditional Neural Process 기반 이미지 인페인팅**:

$$C: \mathbb{R}^m \rightarrow \mathbb{R}^k, \quad O_j \rightarrow C(O_j) = z_j$$

$$\Psi: \mathbb{R}^k \rightarrow \mathbb{R}^l, \quad z_j \rightarrow \Psi(z_j) = \theta_j$$

**CelebA 데이터셋 (32×32) 결과**:

| 문맥 픽셀 수 | CNP | Set Encoder | CNN Encoder |
|-----------|-----|-----------|-----------|
| 10 | 0.039 | 0.035 | **0.033** |
| 100 | 0.016 | 0.013 | **0.009** |
| 1,000 | 0.009 | 0.009 | **0.008** |

하이퍼네트워크를 통한 SIREN의 일반화는 조건부 신경 프로세스(CNP)와 동등하거나 우수하다.[1]

***

## 4. 논문의 한계와 제약 사항

### 4.1 주요 한계점

**1. 계산 복잡도**:
- 도함수 계산이 복잡함 (L계층 네트워크의 기울기는 L(L+1)/2계층 SIREN을 평가하는 것과 동일)
- 깊은 네트워크에서는 상당한 계산 오버헤드 발생[1]

**2. 주파수 설정의 민감성**:
- 첫 번째 계층의 $\omega_0$ 값 선택이 중요함
- 논문에서는 모든 응용에 $\omega_0 = 30$을 사용했으나, 신호 특성에 따라 최적값이 다를 수 있음[1]

**3. 고차원 문제에서의 확장성**:
- 차원이 증가함에 따라 신경망 크기와 학습 복잡도가 지수적으로 증가 (차원의 저주)[1]

**4. 훈련 속도**:
- 일부 응용에서는 여전히 상당한 계산 시간 필요
  - Poisson 방정식: ~90분
  - SDF 학습: ~6시간
  - 파동 방정식: ~25시간[1]

**5. 최적화 어려움**:
- 경계 값 문제에서 초기 조건과 경계 조건의 가중치 균형이 중요
- 수동으로 손실 함수의 하이퍼파라미터를 조정해야 함[1]

***

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 하이퍼네트워크를 통한 일반화

**메타학습 접근**:
SIREN의 모든 매개변수 $\theta_j \in \mathbb{R}^l$이 $k$차원 잠재 벡터 $z \in \mathbb{R}^k$ (단, $k < l$)로 표현 가능하다고 가정하면, 이는 SIREN 함수 공간에 대한 사전분포(prior)를 학습하는 것이다.[1]

**성능 향상**:
- 부분 관찰 $O$에서 추출한 특징으로 조건화된 이미지 인페인팅 가능
- 동일한 모델으로 다양한 관찰 수준에 대응[1]

### 5.2 최신 연구의 일반화 개선 방향

**1. H-SIREN (2024)**:
- 첫 번째 계층 활성화를 $\sin(\sinh(2x))$로 변경
- 다양한 컴퓨터 비전 및 유체 시뮬레이션 작업에서 성능 향상[2]

**2. 적응형 위치 인코딩 (APE)**:
- NeRF와의 결합에서 고정된 주파수 대신 학습 가능한 주파수 대역 도입
- 번들 조정(bundle adjustment) 성능 개선[3]

**3. 메타학습을 통한 분류 (2025)**:
- MAML/Meta-SGD로 SIREN 초기화 및 학습률 스케줄 학습
- CIFAR-10: 38.8% → 59.6% (증강 없음), 63.4% → 64.7% (증강 포함)
- ImageNet-1K에서 23.63% 정확도 달성[4]

**4. 시간 시계열 임퓨팅 (MADS, 2023)**:
- 모듈화된 자동 디코딩 프레임워크로 불규칙한 데이터 처리 향상[5]

### 5.3 일반화 성능 개선의 핵심 요소

**메타학습 기반 접근의 효과**:
$$\min_{\phi} \sum_{T \sim p(\mathcal{T})} \mathcal{L}(\theta - \alpha \nabla_{\theta}\mathcal{L}(D_{\text{train}}), D_{\text{test}})$$

초기화와 학습률을 메타학습하면 적응성이 향상된다.[4]

**주파수 적응**:
- 고정 주파수 대신 학습 가능한 주파수: $\sin(2^b \pi x)$의 $b$ 값을 학습
- 신호 특성에 맞는 표현 공간 자동 구성[3]

---

## 6. 논문의 영향 및 앞으로의 연구 방향

### 6.1 논문이 미친 영향

**인용 수**: 3,658회 이상의 높은 인용도 (2020년 발표 이후)[6]

**주요 후속 연구 분야**:

**1. 신경 방사 필드(NeRF) 발전**:
- SIREN은 NeRF의 위치 인코딩(positional encoding) 개발의 영감 제공
- NeRF는 위치 인코딩으로 유사한 주기적 함수 특성 도입[7]

**2. 암묵적 표현의 광범위한 응용**:
- **의료 이미징**: 뇌 이미지 등록(Brain MRI registration)[8][9]
- **현미경**: 광학 현미경에서 중간 평면 예측 및 모션 아티팩트 보정[10]
- **압축**: 신경 암묵적 이미지 압축 방법 개선[11]
- **분류**: 생 신경망 가중치로부터 직접 분류[4]

**3. 활성화 함수 연구**:
- 기존 ReLU, tanh 기반 방법에서 주기 함수의 우월성 재인식
- 다양한 주기 함수 활성화 실험 (snake, chirp, Morlet 웨이블릿)[8]

**4. PDE 및 과학 계산**:
- Physics-informed Neural Networks (PINNs)와의 유사성 강조
- 미분 방정식 풀이에 신경망의 활용성 확대

### 6.2 앞으로의 연구 고려 사항

**1. 계산 효율성 개선**:
- 도함수 계산의 자동미분 최적화
- 블록 구조적 근사를 통한 계산량 감소
- GPU 병렬 처리의 효율적 활용

**2. 일반화 성능 확대**:
- 도메인 적응 학습 방법 개발
- 다중 신호 동시 표현을 위한 멀티태스크 학습
- 메타학습 프레임워크의 심화 (최근 2025년 연구에서 진행 중)[4]

**3. 고차원 문제 해결**:
- 차원의 저주 극복을 위한 국소 표현 결합
- 계층적 표현 구조 개발
- 희소성(sparsity) 도입

**4. 다양한 주기 함수 활성화 탐색**:
- 신호 특성별 최적 활성화 함수 선택 기준 정립
- 학습 가능한 주파수 파라미터 도입 (APE 등)
- 적응형 초기화 방식 개발

**5. 하이브리드 접근**:
- 암묵적 표현과 명시적 표현(격자 기반)의 결합
- 국소 좌표 시스템과 전역 신경망의 통합
- 다중 스케일 표현 구조

**6. 물리 기반 제약 강화**:
- 경계 조건의 더 정교한 인코딩
- 물리 법칙을 직접 위반하지 않는 손실 함수 설계
- 에너지 보존, 모멘텀 보존 등의 보조 제약

### 6.3 실용적 적용 시 주의점

**1. 신호 특성 분석**:
- 입력 신호의 주파수 성분 사전 분석
- $\omega_0$ 값 설정 시 신호 대역폭 고려

**2. 학습 안정성**:
- 손실 함수 항들의 스케일 균형 (예: Helmholtz 방정식에서 $\lambda(x)$ 설정)
- 충분한 배치 크기 확보로 기울기 노이즈 감소

**3. 메모리 효율성**:
- 고해상도 신호의 경우 국소 배치 샘플링 활용
- 동적 네트워크 구조 고려

**4. 검증 방법론**:
- 도함수 표현 품질 평가 (PSNR뿐만 아니라 도함수 오차도 측정)
- 물리적 제약 만족도 검증

***

## 결론

SIREN은 암묵적 신경 표현 분야에서 **패러다임 전환**을 가져온 논문이다. 주기함수 활성화의 도입과 원리에 기반한 초기화 방식은 ReLU 기반 방법의 한계를 명확히 보여주었으며, 이는 후속 신경 방사 필드(NeRF)와 많은 응용 연구의 기초가 되었다. 향후 메타학습, 적응형 주파수 최적화, 물리 기반 제약 강화 등의 방향에서 지속적인 개선이 기대되며, 특히 계산 효율성과 고차원 확장성은 실제 응용을 위한 중요한 과제로 남아있다.[2][5][10][11][4][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/88a54b8c-5189-4c12-86a3-1e314eddb036/2006.09661v1.pdf)
[2](https://arxiv.org/html/2410.04716)
[3](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_Adaptive_Positional_Encoding_for_Bundle-Adjusting_Neural_Radiance_Fields_ICCV_2023_paper.pdf)
[4](https://openaccess.thecvf.com/content/CVPR2025/papers/Gielisse_End-to-End_Implicit_Neural_Representations_for_Classification_CVPR_2025_paper.pdf)
[5](https://arxiv.org/html/2307.00868)
[6](https://dl.acm.org/doi/10.5555/3495724.3496350)
[7](https://nuggy875.tistory.com/168)
[8](https://pmc.ncbi.nlm.nih.gov/articles/PMC10575995/)
[9](https://www.nature.com/articles/s41598-023-44517-5)
[10](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019677/)
[11](https://arxiv.org/html/2409.17134)
[12](http://arxiv.org/pdf/2006.09661.pdf)
[13](https://arxiv.org/html/2503.18123v1)
[14](https://arxiv.org/abs/2109.00249)
[15](https://github.com/vsitzmann/awesome-implicit-representations)
[16](https://openreview.net/pdf/38d88a65d4e30f1a490e046a1016ab98614834e4.pdf)
[17](https://mole-starseeker.tistory.com/114)
[18](https://arxiv.org/pdf/2503.19576.pdf)
[19](https://proceedings.neurips.cc/paper/2020/file/53c04118df112c13a8c34b38343b9c10-Paper.pdf)
