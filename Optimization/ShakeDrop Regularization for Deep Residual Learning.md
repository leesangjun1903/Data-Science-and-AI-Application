# ShakeDrop Regularization for Deep Residual Learning

**핵심 주장 및 주요 기여**  
본 논문은 ResNet 계열(ResNet, Wide ResNet, PyramidNet, ResNeXt)의 과적합(overfitting) 문제를 완화하기 위해 두-및 세-브랜치 구조 모두에 적용 가능한 새로운 확률적 정규화 기법인 **ShakeDrop**을 제안한다. 주요 기여는 다음과 같다.[1]
- **범용성**: 기존 Shake-Shake 정규화(ResNeXt 전용)와 달리 ResNet, Wide ResNet, PyramidNet 등 두-브랜치 네트워크에도 적용 가능  
- **강력한 일반화 성능**: CIFAR-10/100, ImageNet, COCO 실험에서 RandomDrop, Shake-Shake 대비 일관된 성능 향상  
- **안정성 확보**: RandomDrop(Stochastic Depth)과 Shake-Shake의 장점을 결합해 훈련 불안정성을 제어함으로써 강한 노이즈 주입에도 수렴 보장  

***

## 1. 해결하고자 하는 문제  
딥 네트워크가 매우 깊어질수록 일반화 오차(test error)와 훈련 오차(train error) 간 격차가 커지는 과적합 문제가 심화된다.  
- ResNet 계열의 다양한 변종(ResNeXt, Wide ResNet, PyramidNet)도 여전히 과적합에 취약  
- 기존 정규화 기법(데이터 증강, 드롭아웃, 배치 정규화 등)으로 충분한 일반화 성능 향상 한계  

***

## 2. 제안 방법: ShakeDrop 정규화  
ShakeDrop은 각 residual block 출력을 랜덤 스위칭하여 “원본 네트워크”와 “노이즈 주입 네트워크”를 확률적으로 선택하는 메커니즘을 도입한다.  
### 2.1 수식 정의  
각 블록의 입력 $$x$$, 잔차 함수 $$F(x)$$에 대해  

$$
G(x) = 
\begin{cases}
x + (b_l + \alpha - b_l \alpha)\,F(x), & \text{train-fwd}\\
x + (b_l + \beta  - b_l \beta)\,F(x), & \text{train-bwd}\\
x + \mathbb{E}[\,b_l + \alpha - b_l \alpha\,]\,F(x), & \text{test}
\end{cases}
$$  

- $$b_l \sim \mathrm{Bernoulli}(p_l)$$ : 블록 드롭 확률 $$p_l$$ (layer depth에 따른 선형 감쇠)  
- $$\alpha$$, $$\beta$$ : 독립적 균등 분포 랜덤 변수  
- **train-fwd**: 정방향 계산, **train-bwd**: 역전파 계산  
- $$\alpha,\beta$$ 범위:  
  - EraseReLU 네트워크: $$\alpha\in[-1,1],\;\beta\in$$[1]
  - 일반 네트워크: $$\alpha=0,\;\beta\in$$[1]

### 2.2 메커니즘  
- $$b_l=1$$ (원본 네트워크): $$G(x)=x+F(x)$$로 기존 ResNet과 동일  
- $$b_l=0$$ (노이즈 주입): $$\alpha,\beta$$에 따라  
  - 정방향: $$x+\alpha F(x)$$  
  - 역전파: $$x+\beta F(x)$$  
- **안정화**: Shake-Shake의 양방향 브랜치 안정화 메커니즘과 RandomDrop의 확률적 선택을 결합하여 강한 노이즈에도 수렴 보장  

***

## 3. 모델 구조 및 구현  
- ShakeDrop은 residual block 내 병렬 브랜치 없이 두-브랜치·세-브랜치 네트워크 모두에 삽입 가능  
- 세-브랜치(ResNeXt) 적용 방식  
  - Type A: 브랜치 합산 후 D(·) 적용  
  - Type B: 각 브랜치별 D(·) 적용 후 합산  
- 두-브랜치(ResNet, Wide ResNet, PyramidNet)에는 Single-branch Shake 대신 ShakeDrop 수식 직접 사용  

***

## 4. 성능 향상 및 한계  
### 4.1 CIFAR-10/100  
- ResNet-110: ShakeDrop 23.74% vs RandomDrop 24.07% vs Vanilla 27.42%  
- PyramidNet-110: ShakeDrop 15.78% vs RandomDrop 17.74% vs Vanilla 18.01%[1]
### 4.2 ImageNet  
- ResNet-152: ShakeDrop 20.88% vs RandomDrop 21.33% vs Vanilla 21.72%  
- ResNeXt-152: ShakeDrop 20.34% vs RandomDrop 20.45% vs Vanilla 20.49%  
### 4.3 COCO Detection/Segmentation  
- Faster R-CNN (ResNet-152): AP↑ from 39.0→40.1  
- Mask R-CNN (ResNet-152): AP↑ from 36.6→36.9  
### 4.4 한계 및 민감도  
- 얕은 네트워크(층 수 적음)에서는 p_l 설정 민감도 ↑  
- Wide ResNet-28-10k(얕은 구조): ShakeDrop 수렴 실패 사례 존재  
- 최적 p_l, $$\alpha,\beta$$ 범위는 아키텍처별 실험적 튜닝 필요  

***

## 5. 일반화 성능 향상 가능성  
ShakeDrop은 **피처 공간(feature space)**에 고차원 노이즈 / 스위칭을 반복 장입하여 모델이 지역 최소해(local minima)에 수렴하지 않고 넓은 범위의 매개변수 공간을 탐색토록 한다.  
- Shake-Shake의 데이터 합성(data augmentation in feature space) 효과  
- RandomDrop의 확률적 네트워크 깊이 스위칭 효과  
이 두 기작이 결합되어 다양한 노이즈 조합 하에서도 **견고한 표현 학습**을 유도, unseen 데이터에 대한 **일반화 능력**을 크게 개선한다.[1]

***

## 6. 향후 연구에 미치는 영향 및 고려 사항  
- **범용 정규화 기법**: ResNet 패밀리 외 다른 아키텍처(Transformer, Graph NN 등)에도 ShakeDrop 원칙 적용 가능성  
- **조합 정규화**: Mixup, Manifold Mixup 등 다른 강력한 정규화와의 시너지 연구 필요  
- **이론적 분석**: $$\alpha,\beta$$ 분포와 수렴/최적화 궤적 간 이론적 관계 규명  
- **자동 튜닝**: 아키텍처별 최적 $$p_l,\alpha,\beta$$ 탐색을 위한 메타 학습 또는 Bayesian 최적화 기법 적용  
- **안정화 기법**: 얕은 네트워크에서의 수렴 실패 문제 보완을 위한 적응적 확률 조정 메커니즘 개발  

ShakeDrop은 딥러닝 모델의 **강력한 일반화 정규화** 방안을 제시하여, 범용 정규화 연구 분야에 중요한 토대를 제공한다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f17f0f17-fbcc-40f6-8ca2-d8ce34590fe0/1802.02375v3.pdf)
