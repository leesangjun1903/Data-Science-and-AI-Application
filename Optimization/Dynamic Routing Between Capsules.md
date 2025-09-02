# Dynamic Routing Between Capsules

## 1. 핵심 주장 및 주요 기여  
이 논문은 **벡터 형태의 캡슐(Capsule)** 구조와 **동적 라우팅(routing-by-agreement)** 메커니즘을 도입하여, 기존의 CNN이 갖는 최대 풀링(max-pooling)의 정보 손실 문제를 해결하고, 겹쳐진 객체들에 대한 분할(segmentation) 및 패트런 인식에서 탁월한 성능을 보임을 보였다.[1]

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
- CNN의 최대 풀링은 위치 정보 손실 및 작은 객체나 겹침(overlap) 처리에 취약  
- 객체의 자세(pose) 변화를 일반화하기 위해 대규모 데이터 또는 복잡한 구조 필요  

### 2.2 제안하는 방법  
- 각 캡슐의 **활성화 벡터** 길이를 객체 존재 확률로 사용하고, 방향을 인스턴스화 파라미터로 활용  
- 동적 라우팅 알고리즘:  
  1. 하위 캡슐의 출력 $$\mathbf{u}\_i $$에 가중치 행렬 $$\mathbf{W}\_{ij} $$ 곱하여 예측 벡터 $$\hat{\mathbf{u}}\_{j|i} = \mathbf{W}_{ij} \mathbf{u}_i $$ 생성  
  2. 초기 라우팅 로짓 $$b_{ij}=0 $$에서 시작하여, softmax를 통해 coupling coefficient $$c_{ij} = \frac{e^{b_{ij}}}{\sum_k e^{b_{ik}}} $$ 계산  
  3. 상위 캡슐 입력 $$\mathbf{s}\_j = \sum_i c_{ij} \hat{\mathbf{u}}_{j|i} $$ → squash 함수로 출력 $$\mathbf{v}_j $$ 생성:  

$$
       \mathbf{v}_j = \frac{\|\mathbf{s}_j\|^2}{1 + \|\mathbf{s}_j\|^2} \frac{\mathbf{s}_j}{\|\mathbf{s}_j\|}
     $$  
  
  4. 동적 업데이트: $$b_{ij} \leftarrow b_{ij} + \hat{\mathbf{u}}_{j|i} \cdot \mathbf{v}_j $$  
  5. r회 반복  

- **Margin loss**:  

$$
    L_k = T_k \max(0, m^+ - \|\mathbf{v}_k\|)^2 + \lambda (1 - T_k) \max(0, \|\mathbf{v}_k\| - m^-)^2
  $$  
  
($$m^+=0.9, m^-=0.1, \lambda=0.5$$)

### 2.3 모델 구조  
- Conv1: 256개의 $$9\times9$$ 필터, ReLU  
- PrimaryCapsules: 32채널, 각 채널이 $$6\times6$$ 격자의 8D 캡슐, 스트라이드 2  
- DigitCaps: 10개의 16D 캡슐(각 숫자 클래스별)  
- 재구성 네트워크: 정답 캡슐의 벡터로부터 3개의 완전 연결층을 통해 입력 이미지 복원  

### 2.4 성능 향상  
- MNIST: test error 0.25% (기존 CNN 0.39%)  
- 겹친 숫자(MultiMNIST): error 5.2% (기존 모델 대비 크게 개선)  
- affNIST(affine 변환): CapsNet 79% vs CNN 66%  
- CIFAR-10, smallNORB, SVHN 등 다양한 데이터셋에서 경쟁력 있는 결과  

### 2.5 한계  
- 라우팅 반복 수(routing iteration) 증가 시 과적합 위험  
- 복잡한 자연 배경에서 “orphan” 캡슐 도입 필요  
- 계산 비용 및 메모리 요구량 증가  

## 3. 일반화 성능 향상 가능성  
캡슐의 **벡터 인스턴스화 파라미터** 표현은 다양한 관점(viewpoint) 변화를 행렬 변환으로 자동 일반화하며, 소수의 학습 샘플로도 견고한 표현 학습을 가능케 한다. 특히 affine 변환, 회전, 스케일링 등에 대해 CNN 대비 내재적인 불변성을 보이며, 재구성 손실(reconstruction loss)이 자세 정보 보존을 강화해 일반화 성능을 더욱 높인다.[1]

## 4. 향후 연구에 미치는 영향 및 고려점  
- **연구 방향**: 캡슐 구조의 심층화, 동적 라우팅 효율화, 복잡한 장면 분할 응용  
- **고려 사항**:  
  - 계산 효율화(라우팅 최적화)  
  - 대규모 자연 이미지에서의 잡음·배경 처리 메커니즘  
  - 캡슐 네트워크와 Transformer 등 다른 구조의 융합 가능성  

 attached_file:1[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/96fb3b45-5374-426f-9832-ac6dac2f739b/1710.09829v2.pdf)
