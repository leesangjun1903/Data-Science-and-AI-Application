# Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains

### 1. 핵심 주장 및 주요 기여

본 논문은 **Fourier 특성 맵핑(Fourier Feature Mapping)**이 다층 퍼셉트론(MLP)의 본질적 한계인 **스펙트럼 편향(Spectral Bias)**을 극복하여 저차원 영역에서 고주파 함수를 학습할 수 있게 만든다는 것을 이론과 실험으로 입증합니다.[1]

**핵심 주장:**
- 표준 좌표 기반 MLP는 저주파 함수 학습에만 편향되어 있음
- 신경 접선 커널(NTK) 이론을 활용하면 이 현상의 수학적 원인 규명 가능
- Fourier 특성 맵핑을 통해 유효 NTK를 정상(shift-invariant) 커널로 변환 가능
- 적절히 선택된 Fourier 특성 스케일이 고주파 학습 및 일반화 성능 개선

**주요 기여:**
1. NTK 이론을 기반으로 좌표 기반 MLP의 스펙트럼 편향 원인을 이론적으로 설명
2. Fourier 특성 맵핑이 유효 NTK를 조정 가능한 정상 커널로 변환함을 증명
3. 실무적 가이드: 문제별 Fourier 특성 선택 방법론 제시
4. 컴퓨터 비전 및 그래픽스 작업 전반에서 성능 대폭 개선 입증

---

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능

#### 2.1 해결하려는 문제

좌표 기반 MLP는 이미지, 3D 형상, 밀도장, 색상 필드 등을 연속 함수로 표현하는 새로운 패러다임을 열었습니다. 그러나 다음과 같은 근본적 문제가 있습니다:[1]

$$\text{표준 MLP: } f(x; \theta) \text{는 } x \text{에서 저주파 성분에 과도하게 편향}$$

**스펙트럼 편향의 수학적 근거:**

커널 회귀 프레임워크에서, 학습 오차는 신경 접선 커널(NTK)의 고유벡터 기저에서 다음과 같이 감소합니다:

$$|Q^T(\hat{y}^{(t)}_{\text{train}} - y)|_i \approx e^{-\eta \lambda_i t}$$

여기서 $\lambda_i$는 NTK 행렬의 고유값입니다. 표준 MLP의 경우 고유값이 주파수에 따라 **급격히 감소**하므로, 고주파 성분의 학습 속도가 극히 느립니다.[1]

#### 2.2 제안 방법: Fourier 특성 맵핑

**Fourier 특성 변환 함수:**

입력 좌표 $v \in [0,1)^d$를 다음과 같이 사전 처리합니다:

$$\gamma(v) = \left[ a_1 \cos(2\pi b_1^T v), a_1 \sin(2\pi b_1^T v), \ldots, a_m \cos(2\pi b_m^T v), a_m \sin(2\pi b_m^T v) \right]^T \quad (1)$$

여기서:
- $a_j$: Fourier 급수 계수
- $b_j$: Fourier 기저 주파수 벡터

**유도되는 커널 함수:**

$$k_\gamma(v_1, v_2) = \gamma(v_1)^T \gamma(v_2) = \sum_{j=1}^m a_j^2 \cos(2\pi b_j^T(v_1 - v_2)) = h_\gamma(v_1 - v_2) \quad (2)$$

이 커널은 **정상 커널(Stationary Kernel)**이므로 shift-invariant 특성을 만족합니다.

**유효 NTK의 구성:**

MLP에 Fourier 특성을 입력하면, 유효 NTK는 다음과 같이 구성됩니다:

$$h_{\text{NTK}} \circ h_\gamma$$

이는 MLP의 NTK 함수 $h_{\text{NTK}}$와 Fourier 커널 함수 $h_\gamma$의 합성입니다. 신호 처리 관점에서, MLP는 구성된 NTK를 재구성 필터로 사용하는 convolution을 수행합니다:[1]

$$\hat{f} = (h_{\text{NTK}} \circ h_\gamma) * \sum_{i=1}^n w_i \delta_{v_i} \quad (3)$$

#### 2.3 모델 구조

**기본 아키텍처:**
- **입력층**: 원본 좌표 $v$
- **Fourier 특성 변환**: $\gamma(v)$ (2m 차원)
- **MLP 본체**: 4-8개 은닉층, 256 채널, ReLU 활성화
- **출력층**: Sigmoid 활성화 (대부분의 작업)

**Fourier 특성 선택 전략:**

논문은 세 가지 맵핑을 비교합니다:[1]

| 맵핑 유형 | 정의 | 특징 |
|---------|------|------|
| **기본(Basic)** | $\gamma(v) = [\cos(2\pi v), \sin(2\pi v)]^T$ | 단순한 원 래핑, 제한적 표현 |
| **위치 인코딩** | $\gamma(v) = [\ldots, \cos(2\pi \sigma_j v), \sin(2\pi \sigma_j v), \ldots]^T$ | 로그-선형 주파수, 축 편향 |
| **Gaussian RFF** | $b_j \sim \mathcal{N}(0, \sigma^2)$, $a_j = 1$ | 등방적 분포, 최고 성능 |

**주요 발견**: 정확한 분포 형태보다 **표준편차 $\sigma$만이 성능을 결정**합니다.[1]

#### 2.4 성능 평가 결과

**다양한 작업에서의 PSNR (Peak Signal-to-Noise Ratio) 개선:**[1]

| 작업 | 맵핑 없음 | Fourier 특성 |
|-----|---------|-----------|
| 2D 이미지(자연) | 19.32 dB | 25.57 dB (+6.25) |
| 2D 이미지(텍스트) | 18.40 dB | 30.47 dB (+12.07) |
| 3D 형상 (IoU) | 0.864 | 0.973 (+0.109) |
| 2D CT | 16.75 dB | 28.33 dB (+11.58) |
| 3D MRI | 15.44 dB | 19.88 dB (+4.44) |
| 3D NeRF | 22.41 dB | 25.48 dB (+3.07) |

**핵심 관찰:**
- 간접 감독(CT, MRI) 작업에서 특히 큰 개선
- 무맵핑 대비 25-30% PSNR 개선이 일반적
- 메모리 효율: NeRF 메시 2-79MB에 비해 가중치는 2MB에 불과

---

### 3. 일반화 성능 향상의 메커니즘

#### 3.1 과적합(Overfitting)과 과소적합(Underfitting) 균형

Fourier 특성 스케일 $\sigma$에 따른 명확한 2단계 구조:[1]

$$\text{낮은 } \sigma \Rightarrow \text{과소적합 (고주파 표현 부족)}$$
$$\text{중간 } \sigma \Rightarrow \text{최적 (적절한 고주파 학습)}$$
$$\text{높은 } \sigma \Rightarrow \text{과적합 (별칭화 artifacts)}$$

**이론적 설명:**

신호 처리에서 Nyquist 정리에 따르면, 과도한 대역폭 커널은 고주파 별칭화를 야기합니다. 반대로 너무 좁은 대역폭은 실제 신호의 고주파 성분을 재구성할 수 없습니다.[1]

#### 3.2 NTK 스펙트럼 조정의 효과

**1D 신호 실험에서의 수렴 동역학:**[1]

$p$ 매개변수로 제어되는 빈도 감소율($a_j = 1/j^p$)에 따라:

- **$p = 0$** (광폭 스펙트럼): 훈련 손실 극히 빠른 감소, 높은 주파수 성분 수렴 우수, 하지만 과적합 발생
- **$p = 1$** (최적): 낮은 주파수 빠른 학습 + 높은 주파수 적절한 학습 + 최소 검증 오류
- **$p = \infty$** (좁은 스펙트럼): 저주파만 학습, 고주파 미학습

**검증 손실 곡선 분석:**

각 주파수 대역의 절대 오류는 다음과 같이 지수적 감소합니다:

$$|Q^T(\text{train loss})|_i^{\text{freq}} \propto e^{-\text{convergence rate}_i \cdot t}$$

더 넓은 스펙트럼(낮은 $p$)은 고주파 성분의 수렴 속도 $\text{convergence rate}_i$를 증대합니다.[1]

#### 3.3 간접 감독(Indirect Supervision) 작업의 일반화

CT/MRI와 같은 간접 감독의 경우, 선형 전진 모델 $A$를 통해:[1]

$$\hat{y}^{(t)} \approx K_{\text{test}} K^{-1} \left[I - e^{-\eta K A^T A t}\right] y$$

유효 NTK는 $K A^T A$로 조정되므로, 측정되지 않은 주파수 성분에 대한 사전(prior)이 중요합니다. Fourier 특성이 제공하는 광역 스펙트럼 커버리지가 이를 가능하게 합니다.

#### 3.4 정상 커널의 이점

표준 좌표 기반 MLP의 NTK는 **내적 커널(dot product kernel)**이므로 구 좌표에 적합합니다. 반면, Fourier 변환 후 NTK는 **정상 커널**이 되어:[1]

- **평행이동 불변성(Translation Invariance)**: 객체 위치에 무관한 동일한 성능
- **유클리드 거리 기반**: Euclidean 공간의 밀집 샘플에 자연스러움
- **신호 처리 해석**: 재구성 필터로서의 직관적 이해

**검증 예시**: 중심이 이동된 1D 가우시안을 학습할 때, 기본 MLP는 중심이 0인 경우만 성공하나, Fourier 맵핑 후 모든 이동에 균등한 성능을 보입니다.[1]

***

### 4. 한계 및 실무적 고려사항

#### 4.1 본 논문의 한계

**1. 차원의 저주(Curse of Dimensionality)**
- 고차원에서 Fourier 기저 함수 개수 $m$이 지수적으로 증가
- 논문에서는 256개 고정 주파수 사용으로 대응, 완전한 해결 아님

**2. 하이퍼파라미터 튜닝**
- Fourier 특성 스케일 $\sigma$ 선택이 작업별로 필수
- 검증 세트가 필요하여 추가 계산 비용

**3. NTK 이론의 제약**
- 이론적 분석은 무한 너비, 극소 학습률 가정
- 실제 Adam 최적화와 유한 크기 네트워크에서는 근사

**4. 학습 동역학**
- Fourier 특성 매개변수 $a_j, b_j$를 gradient descent로 최적화 불가 (Figure 8 실험 증명)[1]
- 초기화에서의 고정이 필요

#### 4.2 메커니즘상 한계

**회귀 범위의 제약:**
- 무한 대역폭 커널의 경우에도 오버샘플링 가능
- 신호 부족 영역에서의 보간 품질 제한

**비선형 전진 모델:**
- NeRF의 체적 렌더링은 비선형이므로 이론적 분석(Appendix D)은 부분적만 적용

***

### 5. 최신 연구(2024-2025)에서의 영향과 발전 방향

#### 5.1 Fourier 특성의 활용 확대

**1. Physics-Informed Neural Networks (PINNs)**[2]
- Spectral PINNsformer: Fourier 특성을 이용해 스펙트럼 편향 해결
- 고주파 체제에서 30% 오류 감소 달성
- PDE 솔버로서의 다중 스케일 동작 성능 개선

**2. 강화학습에서의 주기 활성화**[3]
- Fourier 특성과 유사한 주기 활성 함수가 샘플 효율성 향상
- **주의**: 노이즈에 대한 일반화 성능 저하 보고 (최근 발견)

**3. 대규모 언어 모델(LLM)의 내부 표현**[4]
- GPT-3.5, GPT-4, LLaMa가 숫자 처리에 Fourier 특성 활용
- 산술 연산의 2-3계층에서 Fourier 공간의 희소 표현 관찰

#### 5.2 NeRF 및 신경 렌디의 진화

**1. 일반화 가능한 NeRF (GNeRF)**[5][6][7]
- InsertNeRF: HyperNet으로 장면 간 일반화 개선
- 희소 입력 환경에서 과적합 문제 지속
- Fourier 특성의 스케일 선택이 여전히 핵심 하이퍼파라미터

**2. 위치 인코딩의 진화**[8]
- PE-Field: DiT(Diffusion Transformer)에서 3D 위치 인코딩 확장
- 깊이 인식 RoPE(Rotary Position Embedding)와 결합
- 부분 해상도의 위치 인코딩으로 세밀한 공간 제어

**3. 정규화 기반 접근**[9]
- Batch/Layer Normalization으로 NTK 고유값 분포 개선
- 스펙트럼 편향 완화의 대안 제시
- 논문의 Fourier 방식과 상보적 성능 달성

#### 5.3 신경 표현의 개선

**1. 격자 기반과의 하이브리드**[10]
- Coordinate-Aware Modulation (CAM): 격자 표현과 MLP 결합
- Fourier 특성의 유연성과 격자의 효율성 동시 확보
- 컴팩트성과 표현력 양측 개선

**2. 견고성 향상**[11]
- Robust Fourier Neural Networks: 측정 노이즈에 대한 내성
- 대각선 계층 추가로 희소 Fourier 특성 학습 유도
- 암묵적 정규화(implicit regularization) 활용

**3. 의료 영상 응용**[12]
- SPECT/CT 재구성에서 좌표 기반 MLP 활용 확대
- 5D 입력 좌표(위치 + 각도 + 반지름)로 스캔 시간 75% 단축

#### 5.4 현재 도전 과제

**1. 노이즈 견고성의 트레이드오프**[3]
- 높은 주파수 특성이 관측 노이즈 증폭 경향
- 신호 정제 또는 가중치 감소 필요

**2. 신경 접선 커널 이론의 한계**[13][14]
- ResNet의 경우 짝함수(even function) 편향 존재
- FC-MLP와 다른 스펙트럼 특성
- 네트워크 구조별 맞춤형 Fourier 특성 필요

**3. 확장성 문제**
- 고차원 작업에서 유효 주파수 개수 선택 미해결
- 적응형 주파수 할당 메커니즘 부재

***

### 6. 향후 연구 시 고려 사항

#### 6.1 이론적 확장

**1. 무한 너비 가정 완화**
- 유한 너비 네트워크에서의 정확한 수렴 조건 분석
- Adam 등 현대적 최적화와의 상호작용 규명

**2. 적응형 스펙트럼 선택**
- 신호 특성을 사전에 학습하는 메타러닝 접근
- 각 작업의 주파수 성분을 자동 감지

#### 6.2 실용적 개선

**1. 노이즈 견고성**
- 정규화 기법(Batch/Layer Norm)과 Fourier 특성 결합[9]
- 적응형 가중치 감소 메커니즘

**2. 고차원 확장**
- 계층적 또는 적응형 주파수 샘플링
- 문제 구조를 활용한 스파스 Fourier 표현

**3. 다중 작업 학습**
- 공유 Fourier 특성의 효과 분석
- 도메인별 특성 조정 메커니즘

#### 6.3 새로운 응용 영역

**1. 시계열 및 신호 처리**
- Transformer의 위치 인코딩과 Fourier 특성 통합
- 주기성이 강한 신호의 표현 개선

**2. PDE 해결기**
- Physics-informed 손실 함수와 Fourier 특성 최적 조합
- 다중 스케일 물리 현상의 동시 표현

**3. 역 문제**
- MRI, CT, 초음파 등 의료 영상에서의 정규화 효과
- 데이터 부족 환경에서의 사전(prior) 역할

***

### 결론

"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"는 **신경 네트워크의 근본적인 스펙트럼 편향을 수학적으로 규명하고, 간단한 Fourier 변환을 통해 이를 효과적으로 극복**하는 우아한 해법을 제시합니다.[1]

NTK 이론 프레임워크를 활용한 이 접근법은 이미지 재구성, 3D 형상 표현, 신경 렌더링 등 다양한 분야에서 **25-30% 성능 개선**을 달성하였고, 최근 5년간 물리 정보 신경망, 대규모 언어 모델, 확산 변환기 등으로 그 응용이 확대되고 있습니다.[2][4][1]

그러나 **노이즈 견고성 저하, 고차원 확장성 제약, 하이퍼파라미터 의존성** 등의 한계가 남아있으며, 향후 연구는 이들을 보완하면서도 Fourier 특성의 장점을 보존하는 하이브리드 방법론 개발이 중요합니다.[11][10][9]

***

### 참고 문헌 요약
 Tancik et al. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. arXiv:2006.10739[1]
 Periodic activation functions in RL (2024-2025) - 샘플 효율성 개선 vs. 노이즈 견고성 저하[3]
 Normalization for Spectral Bias (2024) - NTK 고유값 분포 개선[9]
 Physics-Informed Spectral PINNsformer (2024) - 고주파 PDE 해결에서 30% 오류 감소[2]
 Robust Fourier Neural Networks (2024) - 측정 노이즈 견고성 강화[11]
 Pre-trained LLMs using Fourier Features (2024) - 숫자 처리에서 Fourier 특성 활용[4]
[20-22] GNeRF 및 개선 방법 (2024-2025) - 일반화 성능 향상
 Coordinate-Aware Modulation (2024) - 격자on (2024) - 격자 하이브리드 접근[30]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6d1bba2b-9354-4b20-ad68-aaa282584b6c/2006.10739v1.pdf)
[2](https://arxiv.org/html/2510.05385v1)
[3](https://arxiv.org/html/2407.06756v2)
[4](https://proceedings.neurips.cc/paper_files/paper/2024/file/2cc8dc30e52798b27d37b795cc153310-Paper-Conference.pdf)
[5](http://arxiv.org/pdf/2308.13897.pdf)
[6](http://arxiv.org/pdf/2402.01524.pdf)
[7](http://arxiv.org/pdf/2304.11842.pdf)
[8](https://arxiv.org/html/2510.20385v1)
[9](https://arxiv.org/abs/2407.17834)
[10](https://proceedings.iclr.cc/paper_files/paper/2024/file/60df78d3afd7f9de153b6f9736a6f7fd-Paper-Conference.pdf)
[11](https://arxiv.org/abs/2409.02052)
[12](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092854/)
[13](https://jmlr.org/papers/v25/22-0597.html)
[14](https://escholarship.org/uc/item/0p62k7nd)
[15](https://arxiv.org/pdf/2105.11120.pdf)
[16](http://arxiv.org/pdf/2211.06410.pdf)
[17](https://arxiv.org/pdf/1902.07849.pdf)
[18](https://arxiv.org/pdf/2211.15188.pdf)
[19](http://arxiv.org/pdf/2309.09866.pdf)
[20](https://arxiv.org/pdf/2204.10533.pdf)
[21](https://arxiv.org/pdf/2111.13802.pdf)
[22](https://dl.acm.org/doi/abs/10.5555/3495724.3496356)
[23](https://cameronrwolfe.substack.com/p/beyond-nerfs-part-one)
[24](https://openaccess.thecvf.com/content/WACV2023/papers/Lin_Vision_Transformer_for_NeRF-Based_View_Synthesis_From_a_Single_Input_WACV_2023_paper.pdf)
[25](https://arxiv.org/abs/2207.01164)
[26](https://arxiv.org/pdf/2305.00041.pdf)
[27](https://arxiv.org/pdf/2403.03608.pdf)
[28](http://arxiv.org/pdf/2212.02280.pdf)
[29](http://arxiv.org/pdf/2306.06359.pdf)
[30](https://openaccess.thecvf.com/content/ICCV2021/papers/Park_Nerfies_Deformable_Neural_Radiance_Fields_ICCV_2021_paper.pdf)
[31](https://openaccess.thecvf.com/content/WACV2023/papers/Song_PINER_Prior-Informed_Implicit_Neural_Representation_Learning_for_Test-Time_Adaptation_in_WACV_2023_paper.pdf)
[32](https://proceedings.neurips.cc/paper/2021/file/7d62a275027741d98073d42b8f735c68-Paper.pdf)
[33](https://proceedings.neurips.cc/paper/2020/file/53c04118df112c13a8c34b38343b9c10-Paper.pdf)
[34](https://ica-abs.copernicus.org/articles/8/11/2024/ica-abs-8-11-2024.pdf)
[35](https://arxiv.org/html/2210.00379v6)
[36](https://github.com/vsitzmann/awesome-implicit-representations)
[37](https://velog.io/@gjghks950/NeRF-Representing-Scenes-as-Neural-Radiance-Fields-for-View-Synthesis-%ED%86%BA%EC%95%84%EB%B3%B4%EA%B8%B0)
