# LayerCAM: Exploring Hierarchical Class Activation Maps for Localization

이 보고서는 **LayerCAM: Exploring Hierarchical Class Activation Maps for Localization** 논문의 핵심 아이디어와 방법론을 자세하고 이해하기 쉽게 설명합니다.

---

## 1. 배경 및 동기

기존의 Class Activation Map (CAM) 기법은 주로 CNN의 최종 합성곱(convolutional) 층에서만 대상을 강조하여 **거칠고(coarse) 낮은 해상도**의 위치 정보를 생성합니다[1].  
- 이는 약한 감독(weakly-supervised) 기반의 물체 위치 추정 및 의미론적 분할(segmentation)에서 픽셀 수준의 정밀도가 요구될 때 한계를 드러냅니다[1].  
- 예시: VGG16의 마지막 층에서 생성된 CAM은 말 전체의 위치는 잡지만, 다리나 귀와 같은 세부 정보는 놓칩니다[1].

---

## 2. LayerCAM의 핵심 아이디어

LayerCAM은 **다양한 깊이의 층(layer)에서 CAM을 생성**하여  
- **거시적(coarse) 위치 정보**: 깊은 층에서 얻음  
- **미시적(fine) 세부 정보**: 얕은 층에서 얻음  
을 결합함으로써 보다 정밀한 객체 위치를 얻는 방법입니다[1].

---

## 3. 방법론

### 3.1. 위치별 가중치(Location-specific Weighting)
- 기존 Grad-CAM은 채널별(global) 가중치 $$w_k$$를 모든 공간 위치(i, j)에 동일하게 적용합니다[2].  
- LayerCAM은 각 위치(i, j)에 대한 **개별 가중치**를 활용합니다:
  
$$w^c_{k, ij} = \mathrm{ReLU}\bigl(\tfrac{\partial y_c}{\partial A^k_{ij}}\bigr),$$  

  여기서 $$y_c$$는 클래스 $$c$$에 대한 예측 점수, $$A^k_{ij}$$는 k번째 특징 맵의 위치(i, j) 값입니다[1].

### 3.2. 가중 특징 맵 생성
- 각 특징 맵 $$A^k$$의 위치별 값에 가중치를 곱합니다:

$$\hat{A}^k_{ij} = w^c_{k, ij}\,\cdot A^k_{ij}.$$

### 3.3. CAM 생성
- 가중 특징 맵 $$\hat{A}^k$$들을 채널 축으로 합산하고 ReLU를 적용하여 최종 CAM $$M^c$$을 얻습니다:

$$M^c = \mathrm{ReLU}\Bigl(\sum_k \hat{A}^k\Bigr).$$

### 3.4. 계층적 CAM 융합(Hierarchical Fusion)
1. **얕은 층**(fine): 공간 해상도가 크지만 값 범위가 작음.  
2. **깊은 층**(coarse): 공간 해상도는 작으나 값 범위가 큼.  
- 얕은 층의 CAM들은 **tanh 스케일 함수**로 정규화하여 깊은 층 CAM과 범위를 맞춘 뒤,  
- **요소별 최대값(element-wise max)** 연산으로 융합합니다[1].  
- 스케일링 식 예시:

$$\tilde{M}^c = \tanh\Bigl(\gamma\ * \frac{M^c}{\max(M^c)}\Bigr)$$

---

## 4. 성능 평가

### 4.1. 약한 감독 기반 물체 위치 추정(WSOL)
| 방법          | loc1 (%) | loc5 (%) |
|---------------|----------|----------|
| Grad-CAM      | 43.62    | 53.99    |
| Grad-CAM++    | 45.44    | 56.42    |
| ScoreCAM      | 39.51    | 49.63    |
| LayerCAM      | **47.24**| **58.74**|
*ILSVRC validation set 기준[1][3].*

- **얕은 층 CAM 융합**으로 loc1, loc5 모두 기존 기법 대비 1.8–4.4%p 향상[1].

### 4.2. 이미지 차폐(Image Occlusion)
- CAM에서 강조된 영역을 차폐 시, 모델 예측 정확도 하락폭이 LayerCAM이 가장 큼:  
  - Top-1: 48.26% (LayerCAM) vs. 50.36% (Grad-CAM)  
  - Top-5: 73.43% vs. 75.62%  
  - 신뢰도(ground-truth 예측 점수): 48.12% vs. 50.24%[1].  
- 이는 LayerCAM이 더 중요한 영역을 잘 포착함을 의미[1].

### 4.3. 산업용 결함 위치 검출
- DAGM-2007 데이터셋에서 ResNet50 기반 분류기 + LayerCAM 적용:  
  - mIoU: 27.26% (LayerCAM) vs. 6.46% (Grad-CAM++) vs. 0.35% (Grad-CAM)  
  - 실시간 처리(~60 FPS) 가능[1].

### 4.4. 약한 감독 기반 의미 분할(WSSS)
- PASCAL VOC 2012 validation set 기준 mIoU 비교:  
  | 방법               | mIoU (%) |
  |--------------------|---------:|
  | Grad-CAM (seed)    |     42.00|
  | LayerCAM (seed)    |   **56.35**|
  - 향상된 CAM 융합으로 의사(pseudo) 라벨 품질이 높아져, 기존 WSSS 알고리즘 성능도 크게 향상됨[1].

---

## 5. 결론

LayerCAM은 간단하지만 **다양한 층에서 CAM을 생성·융합**함으로써  
- **정밀도**: 세부 객체 영역 포착 강화  
- **범용성**: 별도 네트워크 수정 없이 기존 CNN 모델에 적용 가능  
- **효율성**: 실시간 처리 수준 유지  
를 모두 달성한 기법입니다[1].  
다양한 약한 감독 과제와 산업용 결함 검출까지 폭넓게 활용될 수 있습니다.

[1] https://ieeexplore.ieee.org/document/9462463/
[2] https://ieeexplore.ieee.org/document/10556528/
[3] https://ieeexplore.ieee.org/document/10030472/
[4] https://link.springer.com/10.1007/978-3-031-43990-2_30
[5] https://peerj.com/articles/cs-622
[6] https://ieeexplore.ieee.org/document/9098436/
[7] https://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf
[8] https://github.com/PengtaoJiang/LayerCAM-jittor
[9] https://dl.acm.org/doi/10.1109/tip.2021.3089943
[10] https://scispace.com/papers/layercam-exploring-hierarchical-class-activation-maps-for-einima0f0r
[11] https://paperswithcode.com/paper/layercam-exploring-hierarchical-class
[12] https://mmcheng.net/mftp/Papers/21TIP_LayerCAM.pdf
[13] https://pubmed.ncbi.nlm.nih.gov/34156941/
[14] https://www.semanticscholar.org/paper/7a44f1af275c2e07844f0f1ddf4babe70fb821cb
[15] https://linkinghub.elsevier.com/retrieve/pii/S0923596524000511
[16] https://link.springer.com/10.1007/s11063-023-11335-9
[17] https://link.springer.com/10.1007/s11042-020-09556-4
[18] https://dl.acm.org/doi/10.1109/TIP.2021.3089943
[19] https://arxiv.org/pdf/2309.14304.pdf
[20] https://github.com/jacobgil/pytorch-grad-cam
