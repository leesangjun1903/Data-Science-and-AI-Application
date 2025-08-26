# ZeroQ: A Novel Zero Shot Quantization Framework

## 1. 핵심 주장 및 주요 기여  
ZeroQ는 원본 훈련 데이터에 전혀 접근하지 않고도 신경망을 저비트로 양자화할 수 있는 **제로샷(zero-shot) 양자화** 기법을 제안한다.  
- 배치 정규화(BatchNorm) 계층의 통계값을 활용해 가상의 **Distilled Data**를 생성함으로써, 데이터 없이도 양자화 감도(sensitivity) 측정과 활성화 클리핑 범위 탐색이 가능하다.  
- 페어토 프론티어(Pareto frontier) 기반의 자동 **혼합 정밀도(mixed-precision) 비트 설정** 방법을 제안하여, 모델 크기 제약 하에서 최적의 레이어별 비트를 결정한다.  
- 다양한 네트워크(ResNet, MobileNetV2, ShuffleNet, InceptionV3, RetinaNet 등)에 대해 최대 1.71%p 이상의 정확도 향상을 달성하며, 전체 양자화 과정이 30초 이내에 완료되는 매우 낮은 연산 오버헤드를 보인다.  

## 2. 문제 정의 및 해결 방법  
### 문제 정의  
- **데이터 접근 제한**: 민감하거나 독점적인 데이터의 경우, 후처리 양자화(post-training quantization)나 양자화 인지 훈련(quantization-aware fine-tuning)에 필요한 원본 데이터를 사용할 수 없음.  
- **성능 저하**: 데이터 없이 수행되는 기존 제로샷 양자화는 초저비트(≤6bit)에서 정확도가 크게 떨어짐.  
- **혼합 정밀도 탐색 비용**: 레이어별 비트를 조합하는 탐색 공간이 $$m^L$$으로 지수적으로 커 탐색 부담이 큼.  

### Distilled Data 생성  
배치 정규화 통계(평균 $$\mu_i$$, 분산 $$\sigma_i$$)를 이용해 입력 $$\mathbf{x}_r$$를 최적화하여, 각 레이어 활성화의 통계 $$(\tilde\mu_i,\tilde\sigma_i)$$가 원본 통계와 일치하도록 함.  

```math
\min_{\mathbf{x}_r}\sum_{i=0}^{L}\bigl\|\tilde\mu_i-\mu_i\bigr\|_2^2 + \bigl\|\tilde\sigma_i-\sigma_i\bigr\|_2^2
```

이로써 생성된 Distilled Data는 실제 훈련 데이터에 근접한 모델 감도 평가를 가능케 한다.  

### 양자화 감도 측정  
각 레이어 $$i$$를 $$k$$-bit로 양자화했을 때의 출력 분포 차이를 KL 발산으로 측정한다:  

$$
\Omega_i(k) = \frac{1}{N}\sum_{j=1}^N \mathrm{KL}\bigl(M(\theta; x_j)\|M(\tilde\theta_i(k\text{-bit});x_j)\bigr)
$$

여기서 $$x_j$$는 Distilled Data.  

### 혼합 정밀도 비트 설정 (Pareto Frontier)  
목표 모델 크기 $$S_{\mathrm{target}}$$ 제약 아래, 전체 감도 $$\sum_i\Omega_i(k_i)$$를 최소화하는 레이어별 비트 $$k_i$$를 선택:  

$$
\min_{\{k_i\}}\sum_{i=1}^L \Omega_i(k_i)\quad\text{s.t.}\quad\sum_{i=1}^L P_i\,k_i \le S_{\mathrm{target}}
$$

다이나믹 프로그래밍으로 페어토 프론티어를 구성하여, 모델 크기–감도 트레이드오프 상 최적점을 자동 탐색한다.  

## 3. 모델 구조 및 성능  
- **지원 모델**: ResNet18/50/152, MobileNetV2, ShuffleNet, SqueezeNext, InceptionV3, RetinaNet-ResNet50 (COCO).  
- **연산 비용**: ResNet50 양자화 전체 30초(8×V100 1대 기준), Distilled Data 생성 3s, 감도 계산 12s, 페어토 최적화 14s.  
- **주요 성능**:  
  - ResNet50 8/8bit: 정확도 77.67% (0.05%p↓), 6/6bit: 77.43% (OCS 대비 +2.63%p)  
  - MobileNetV2 8/8bit: 72.91% (DFQ 대비 +1.71%p), 6/6bit: 72.85% (Integer-Only 대비 +1.95%p)  
  - RetinaNet COCO W4A8(혼합설정): mAP 33.7 (FQN 대비 +1.2)  
- **한계**:  
  - BatchNorm이 없는 계층 감도 측정은 BN 통계 활용이 어려워, FPN 등 특정 구조에선 약간의 추가 처리 필요.  
  - 페어토 최적화는 독립성 가정에 기반하므로 전역 최적에 대한 이론적 보장은 없음.  

## 4. 일반화 성능 향상 가능성  
Distilled Data는 원본 데이터 분포의 통계를 효과적으로 재구성하여, 모델이 보지 못한 실제 입력에도 유사한 활성화 특성을 보여준다. 이는:  
- **다양한 구조**: 분류뿐만 아니라 물체 검출(Det.)에도 적용 가능함이 입증되었고, 다른 비전 태스크(분할, 검출 등)로 확장이 용이하다.  
- **데이터 편향 대응**: 원본 데이터의 레이어별 통계만 필요하므로, 도메인 이동(domain shift) 상황에서도 BN만 재계산되면 새로운 도메인에 대한 제로샷 양자화가 가능하다.  

## 5. 향후 연구에의 영향 및 고려사항  
ZeroQ는 **데이터 프라이버시가 중요한 엣지 컴퓨팅** 및 **모델 서비스 환경**에서 데이터 없이 빠른 배포를 지원한다.  
- **후속 연구 제언**:  
  - BN 없는 모델(예: LayerNorm 또는 NormFree)용 Distilled Data 생성 기법 연구  
  - 비독립성 가정을 완화한 전역 감도 최적화 또는 효율적 탐색 알고리즘 개발  
  - 생성된 Distilled Data 품질을 높이기 위한 추가 통계(예: 채널별 왜도, 첨도) 활용 방안  
- **고려점**:  
  - 실제 엣지 디바이스에서의 연산 시간 및 메모리 제약을 반영한 경량화  
  - 통계 왜곡이 심한 경우, Distilled Data 최적화 안정성 확보 전략 확보  

ZeroQ는 데이터 없는 상황에서도 **정확도와 효율성을 모두 만족**하는 양자화 기법으로, 향후 엣지 AI와 프라이버시 보존 모델 최적화 분야에 중대한 영향을 미칠 전망이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8f53efeb-d2cb-476b-9d35-f51c8ae68b18/2001.00281v1.pdf)
