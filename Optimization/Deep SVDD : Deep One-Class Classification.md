# Deep SVDD : Deep One-Class Classification

## 주요 주장 및 기여
**Deep SVDD(Deep Support Vector Data Description)**를 통해 이상치(Anomaly) 탐지를 위한 완전하게 딥러닝 기반의 **일류(one-class) 분류** 손실 함수를 제안한다.  
- 기존 딥 AD(Anomaly Detection) 방식들이 주로 재구성 오차(reconstruction error)에 의존한 반면, Deep SVDD는 **하이퍼스피어 최소 부피 학습** 목표를 직접 최적화하여 표현 학습과 이상치 탐지를 동시에 수행  
- ν-Property(상위 ν 비율의 이상치 허용) 이론적 보장 및 `무편향(bias)` 항 제거, ReLU 활성화 사용 등 네트워크 구조상의 **하이퍼스피어 붕괴 방지 전략** 제시  
- MNIST, CIFAR-10, GTSRB 정지신호 데이터셋에서 기존 얕은(shallow)·딥 AD 기법 대비 우수한 AUC 성능 달성  

## 1. 해결하고자 하는 문제
- **고차원·레이블 없는(unlabeled) 데이터** 환경에서 “정상(normal)” 클래스만을 학습하여, 새로운 입력이 정상 분포를 벗어나는지를 판단하는 **One-Class Classification** 문제  
- 기존 One-Class SVM/SVDD는 커널 기반 커널 매트릭스 연산의 계산·기억 복잡도가 높으며, 특징(feature) 설계가 필요  
- Autoencoder·GAN 기반 딥 AD는 **재구성 손실**을 이용하나, 압축 비율 설정의 어려움과 직접적인 이상 탐지 목적 부재  

## 2. 제안하는 방법
### 2.1 Soft-Boundary Deep SVDD 목표식
네트워크 φ(x;W)를 통해 입력 x를 p차원 특징 공간 F로 매핑하여, 중심 c ∈F, 반지름 R을 갖는 최소 부피 하이퍼스피어 안에 정상 샘플을 밀집하도록 학습  

$$
\min_{R,W}R^2 + \frac{1}{\nu n}\sum_{i=1}^n \max\{0,\|\phi(x_i;W)-c\|^2 - R^2\} + \frac{\lambda}{2}\sum_{\ell=1}^L \|W^\ell\|_F^2
$$

- ν∈(0,1]: 허용 이상치 비율 제어(ν-property)  
- λ: 가중치 감쇠 정규화  
- “max” 항: 반지름 밖 표현에 페널티 부여  

### 2.2 One-Class Deep SVDD (단순화 버전)
훈련 데이터가 모두 정상이라 가정하고, 모든 표현과 중심 간 거리 제곱의 평균을 최소화  

$$
\min_{W}\frac{1}{n}\sum_{i=1}^n \|\phi(x_i;W)-c\|^2 + \frac{\lambda}{2}\sum_{\ell=1}^L \|W^\ell\|_F^2
$$

### 2.3 최적화 및 구조적 설계
- **편향(bias) 제거** 및 **ReLU** 활성화 사용: 네트워크가 상수 함수로 수렴(하이퍼스피어 붕괴)하지 않도록 구조적 제약  
- c는 초기 전방향 전달된 출력들의 평균으로 고정  
- SGD(또는 Adam)로 W,(R)를 교대로 최적화  
- 이상치 점수: $$s(x)=\|\phi(x;W^*)-c\|^2$$ (Soft-boundary는 $$s(x)-R^2$$)  

## 3. 모델 구조
- MNIST: LeNet 계열 CNN, 2개의 컨볼루션 모듈 → 최종 Dense 32  
- CIFAR-10: LeNet 확장형, 3개의 컨볼루션 모듈(32→64→128) → Dense 128  
- GTSRB: 3개 모듈(16→32→64) → Dense 32  
- 배치 정규화, Leaky ReLU(α=0.1), weight decay λ=10⁻⁶, 학습률 단계적 스케줄  

## 4. 성능 향상 및 한계
### 4.1 성능
- MNIST One-Class AUC 평균 97–99% 달성, 기존 OC-SVM·KDE·IF 대비 우수  
- CIFAR-10 클래스별 AUC 60–94%로 전반적 경쟁력 확보하였으나, “글로벌 구조”가 뚜렷한 클래스(Airplane, Deer 등)에서는 KDE가 더 우수  
- GTSRB Adversarial 감지 AUC 80%로 DCAE·OC-SVM 대비 개선  

### 4.2 한계
- 네트워크 구조 선택(CNN 필터 크기·깊이)이 **데이터 특성**에 민감  
- Soft-boundary ν 조정 필요, 이상치 비율 사전 추정 어려움  
- 소규모 데이터셋(GAN) 학습에 제약  

## 5. 일반화 성능 향상 가능성
- **하이퍼스피어 중심 c 고정 전략**이 단순하며, 데이터 분포 변화에 취약. 동적 중심 추정이나 attention 구조 도입으로 적응력 향상 가능  
- **다중 하이퍼스피어/혼합 모델**로 다중 정상 모드에 대응하여 복합 분포에 대한 일반화 강화  
- **특징 표현 학습**을 위한 사전 학습(pre-training)과 전이학습(transfer learning) 결합 시 소규모 데이터셋에서도 강건성 확보  

## 6. 향후 연구 영향 및 고려사항
- **Fully deep one-class objective** 제안은 이상 탐지 분야에 새로운 패러다임을 제시하며, 재구성 기반 방법과의 하이브리드, GAN·비지도 표현 학습 결합 연구 촉진  
- **구조적 붕괴 방지** 분석(편향·활성화) 결과는 다른 비지도 이상 탐지 모델 설계 시 유용  
- 대규모·다양 분포 환경에서 ν 조정 및 동적 중심 학습 방안, 멀티태스크나 멀티모달 통합으로 **실제 애플리케이션 적용성** 연구 필요  
- **안정적 임계값 결정**과 적응형 이상치 비율 추정 기법이 후속 연구에서 고려되어야 하며, 실시간 이상 탐지 시스템 통합 관점으로 연속적 학습·메타러닝 접목이 기대됨.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/fb5e1a07-8fb9-488b-89b3-d010d11a06d5/ICML2018.pdf
