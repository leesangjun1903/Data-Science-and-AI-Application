# Progressive Neural Architecture Search | NAS

## 1. 핵심 주장 및 주요 기여  
**Progressive Neural Architecture Search (PNAS)**는 복잡한 신경망 구조 탐색을 기존 강화학습(RL) 기반 NAS 대비 더 효율적으로 수행하는 방법을 제안한다.  
주요 기여:  
- **점진적 탐색(Progressive Search)**: 블록 수가 적은 간단한 셀(cell)부터 시작해 점차 복잡도를 높이며 탐색 범위를 확장.  
- **성능 예측 서로게이트 모델(Surrogate Predictor)**: 훈련된 셀의 검증 정확도를 예측하는 경량 RNN 또는 MLP 앙상블을 도입해, 모든 자식 후보를 실제 훈련 없이 평가하고 유망한 K개만 선택.  
- **컴퓨팅 효율성 극대화**: 모델 평가 횟수를 최대 5배, 전체 학습 예제 수 기준으로는 최대 8배까지 절감하며, CIFAR-10·ImageNet에서 기존 동급 성능 달성.

***

## 2. 문제 정의 및 제안 방법  
### 2.1 문제 정의  
기존 NAS 방법은  
- 거대한 탐색 공간(10^28)에서 강화학습·진화 알고리즘으로 전체 셀 또는 CNN 구조를 직접 탐색  
- 수만 개의 후보 모델을 훈련·평가하므로 연산 비용이 막대  

### 2.2 PNAS 방법 개요  
1. **셀(cell) 기반 탐색 공간**:  
   - 각 셀은 B개의 블록(block)으로 구성. 블록당 두 입력→연산(합성곱·풀링·identity 등)→결합(덧셈)으로 정의.  
   - B≤5일 때 유니크 셀 수 ≈10^12  

2. **점진적 탐색 알고리즘**  
   - b=1부터 시작하여 가능한 모든 셀 S₁을 생성·훈련·평가  
   - b단계 후보 S′ₙ = expand(Sₙ₋₁)를 통해 셀 깊이를 1 증가  
   - **서로게이트 모델** π를 이용해 S′ₙ 전체를 점수화 → 상위 K개 Sₙ 선별 → 실제 훈련·평가 → π 재학습  
   - b=B일 때 최종 상위 1개 셀 반환  

3. **서로게이트 모델(π)**  
   - **입력**: 셀을 블록별 토큰(I₁, I₂, O₁, O₂)으로 인코딩한 시퀀스  
   - **구조**:  
     - RNN 버전: LSTM → fully-connected → sigmoid  
     - MLP 버전: 각 블록 임베딩 평균 → fully-connected  
   - **학습**: 이전 단계에서 얻은 (구조, 검증정확도) 쌍으로 L1 손실 최적화  
   - **앙상블**: 5개 모델로 분할 학습하여 예측 안정성 강화  

### 2.3 주요 수식  
- 블록 수 b에서 가능한 블록 구조 수:  
  $$|B_b| = |I_b|^2 \times |O|^2 \times |C|,\quad |I_b| = b+1,\ |O|=8,\ |C|=1. $$  
- 최종 셀 탐색 대상 수:  
  $$\sum_{b=1}^B |S_b| = |B_1| + (B-1)\times K, \quad K=256,\ B=5\ \Rightarrow\ 136 + 4\cdot256 = 1160. $$

***

## 3. 모델 구조  
- **단일 셀 구조**:  
  - 최대 5개의 블록, 각 블록은 깊이별 분리 합성곱(3×3,5×5,7×7…), 풀링, identity 등 연산  
  - 결합 연산은 elementwise addition만 사용  
- **CNN 구성**:  
  - 선택된 셀을 F 필터로 N회 반복(unroll)  
  - stride-2 셀로 공간 해상도 절반 축소 시 필터 수 2배 증가  
  - 최상단 global average pooling + softmax  

***

## 4. 성능 향상  
### 4.1 탐색 효율  
- CIFAR-10 validation에서 상위 1개 모델 도달 시 PNAS는 1,160개 모델 평가, NAS는 5,808개 평가 → **5배 효율**  
- 전체 예제 처리 수 기준으로는 **8배 가량 절감**  

### 4.2 분류 성능  
- **CIFAR-10**:  
  - PNASNet-5 (B=5,N=3,F=48): test error 3.41% ±0.09%, 파라미터 3.2M → NASNet-A와 동등 성능, 컴퓨팅 21× 절감  
- **ImageNet (Mobile)**:  
  - PNASNet-5 (224×224,≈588M mult-adds): Top-1 74.2%, Top-5 91.9% → MobileNet·ShuffleNet 대비 +3.3%p 이상  
- **ImageNet (Large)**:  
  - PNASNet-5 (331×331,≈25B mult-adds): Top-1 82.9%, Top-5 96.2% → SENet·NASNet-A와 동등 혹은 소폭 상회  

***

## 5. 한계 및 일반화 성능  
- **서로게이트 예측 한계**: 셀 크기 extrapolation 시 rank correlation 0.4–0.5 수준  
- **탐색 공간 제약**: 단일 셀 구조만 탐색, 복합 모듈(다중 셀 타입) 도입 미비  
- **일반화 가능성**: CIFAR→ImageNet 전이 연구에서 상관계수 0.727로 강한 연관성 확인. 그러나 다른 태스크(물체 탐지, 분할)로 확장성 검증 필요  

***

## 6. 향후 연구 방향 및 고려 사항  
- **서로게이트 모델 개선**: Gaussian process, 트러스트-리전 강화 기법 도입  
- **조기 종료(Early Stopping) 적용**: 비유망 모델 학습 조기 중단으로 추가 효율화  
- **획득 함수(Acquisition Function)**: 베이지안 최적화 프레임워크로 expected improvement/UCB 활용  
- **다중 셀 구조 탐색**: Normal/Reduction 셀 구분 복원 혹은 구조별 최적화  
- **다양한 태스크 실험**: ImageNet 이외의 대규모 분류, 객체 탐지, 세그멘테이션 등 일반화 검증  

PNAS는 NAS 연구에 **효율성과 실용성**을 동시에 향상시킨 중요한 이정표로, 향후 NAS 방법론이 보다 경량화되고 폭넓은 태스크로 확장되는 방향을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/daade1c7-4a6a-4793-a688-bcbbaac3d663/1712.00559v3.pdf
