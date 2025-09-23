# BAM : Bottleneck Attention Module | Image classification, Object detection

**핵심 주장**  
BAM은 일반적인 CNN의 모든 피드포워드 모델에 경량으로 통합 가능한 주의 모듈로, 채널 및 공간 축을 별도 경로로 처리하여 병목(bottleneck) 지점에서 효과적인 계층적 주의(attention)를 학습함으로써 표현력을 크게 향상시킨다.

**주요 기여**  
1. **범용적이고 경량화된 주의 모듈 제안**  
   – 채널 및 공간 축 두 경로를 분리 설계하여 파라미터·연산 오버헤드를 최소화.  
2. **병목 지점에의 통합 설계**  
   – 네트워크 다운샘플링 병목마다 BAM을 삽입하여 계층적 주의 구조 형성.  
3. **다양한 모델·데이터셋에서의 성능 검증**  
   – CIFAR-100, ImageNet-1K, VOC 2007, MS COCO 전반에서 classification 및 detection 성능 일관성 있게 개선.  
4. **모듈 일반화 및 과적합 방지**  
   – Reduction ratio 최적화(r=16)와 dilation(d=4)를 통해 경량 모듈이지만 과적합 없이 네트워크 용량을 효율적으로 확장.

***

# 1. 문제 정의 및 제안 방법

## 1.1 해결하고자 하는 문제  
CNN 백본(backbone) 아키텍처가 깊어지거나 넓어지며 표현력이 향상되지만,  
- 단순 레이어 추가 방식은 과적합 위험↑  
- 기존 주의 모듈은 무거운 연산·파라미터 오버헤드 또는 채널·공간 정보 중 하나만 활용  

## 1.2 BAM 모듈 구조 및 수식  
BAM은 입력 특징 맵 $$F \in \mathbb{R}^{C\times H\times W}$$로부터 3D 주의 맵 $$M(F)\in \mathbb{R}^{C\times H\times W}$$을 추론해 다음과 같이 특징을 정제함:  

$$
F' = F + F \,\otimes\, M(F)
$$  

– $$\otimes$$: 요소별 곱  
– 잔차 연결을 통한 그래디언트 흐름 강화  

### 채널 주의 분기  
1. 전역 평균 풀링: $$\mathrm{AvgPool}(F)\to F_c\in\mathbb{R}^{C\times1\times1}$$  
2. MLP:  
   
$$
   M_c(F) = \mathrm{BN}\bigl(W_1\,(W_0\,F_c + b_0)+b_1\bigr)\quad\text{with }W_0\in\mathbb{R}^{\frac C r\times C}, W_1\in\mathbb{R}^{C\times\frac C r}
   $$  
   
– $$r=16$$로 채널 경감  

### 공간 주의 분기  
1. 1×1 Conv 채널 축 경감: $$C \rightarrow \tfrac C r$$  
2. 두 단계 3×3 dilated convolution (dilation $$d=4$$)로 문맥 정보 집계  
3. 1×1 Conv로 1채널 출력 및 BN:
   
$$
   M_s(F) = \mathrm{BN}\bigl(f^{1\times1}_2\bigl(f^{3\times3}_1\bigl(f^{3\times3}_0(F)\bigr)\bigr)\bigr)
   $$  

### 주의 맵 결합  
채널·공간 주의 맵을 $$\mathbb{R}^{C\times H\times W}$$으로 확장 후 element-wise summation:  

$$
M(F)=\sigma\bigl(M_c(F) + M_s(F)\bigr)
$$  

– sigmoid $$\sigma$$ 적용, summation이 가장 안정적 학습 및 성능 우수

***

# 2. 모델 구조와 성능 향상

## 2.1 네트워크 통합 위치  
- ResNet 계열의 병목(bottleneck) 블록 마지막 다운샘플링 지점  
- 병목마다 BAM 삽입으로 **계층적 주의** 형성  
- 병목 위치에만 경량 모듈 추가하여 전체 파라미터·FLOPs 오버헤드 최소화

## 2.2 CIFAR-100·ImageNet 성능 개선  
| Architecture                | Params▲    | GFLOPs▲ | Error/CIFAR-100↓ (%) | Top-1/ImageNet↓ (%) |
|-----------------------------|-----------:|--------:|----------------------:|--------------------:|
| ResNet50 baseline           | 23.71M     | 1.22    | 21.49                 | 24.56               |
| + BAM                       | 24.07M(+0.36) | 1.25(+0.03) | **20.00**          | **24.02**           |
| ResNeXt29 8×64d baseline    | 34.52M     | 4.99    | 18.18                 | –                   |
| + BAM                       | 34.61M(+0.09) | 5.00(+0.01) | **16.71**          | –                   |

– CIFAR-100 에러율 최대 1.49%p 감소  
– ImageNet top-1 약 0.5%p 개선  

## 2.3 Detection 성능 개선  
| Task            | Backbone+Detector           | mAP@0.5 (%) |
|-----------------|-----------------------------|-------------:|
| VOC 2007 SSD    | VGG16 + StairNet            | 78.9         |
| + BAM           | VGG16 + StairNet + BAM      | **79.3**     |
| MS-COCO Faster-RCNN | ResNet101              | 48.4         |
| + BAM           | ResNet101 + BAM             | **50.2**     |

– COCO mAP@0.5 약 1.8%p, VOC 2007 mAP@0.5 0.4%p 상승

***

# 3. 일반화 성능 향상 관점

- **경량 모듈**이지만 다양한 모델·데이터셋에서 일관된 성능 상승  
- 채널·공간 주의를 결합해 *what*과 *where* 양 축에서 특징을 효과적으로 강조  
- 병목 지점에서 계층적 주의 형성이 과적합 억제 및 추론 성능 향상에 기여  
- Squeeze-and-Excitation 대비 파라미터 절감(only bottleneck 적용)에도 동등 이상 효과

***

# 4. 제약 및 한계

- Reduction ratio·dilation 최적값 고정(r=16, d=4)이 다른 도메인에서 반드시 최적은 아님  
- 주의 모듈이 downsampling 지점에만 작동하므로, 해상도 등 세부 공간 정보가 중요한 태스크에는 추가 설계 필요  
- 극도로 경량화된 모바일 모델에는 모듈 경감 전략(채널 경감, 양자화 등) 추가 고려 필요

***

# 5. 향후 연구에 미치는 영향 및 고려사항

**영향**  
- *계층적 주의* 개념을 일반 CNN에 경량으로 확장함으로써, 다양한 비전 과제의 핵심 모듈로 자리잡을 수 있음  
- 병목 지점 집중 설계가 모델 효율성 극대화 방향의 중요한 설계 패턴 제시  

**고려사항**  
- 다양한 해상도·해상력 태스크(예: 세그멘테이션)에 맞춘 다중 스케일 공간 주의 확장  
- 언어·음성 등 멀티모달 도메인에서의 병목 주의 모듈 적용 가능성 및 구조 변형  
- AutoML·NAS 기반 모델 탐색 과정에 BAM 통합으로 자동으로 최적 위치·하이퍼파라미터 학습  

앞으로 BAM의 단순·경량 설계를 기반으로, **다양한 태스크·아키텍처**에 대한 확장과 **모듈 경감**, **자동화된 최적화** 연구가 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/420de742-14b9-487f-bad9-2c0d9a6bed8f/1807.06514v2.pdf
