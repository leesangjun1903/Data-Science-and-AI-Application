# ROI (Region of Interest) 가이드

핵심은 ROI의 좌표 체계, 넘파이 슬라이싱 규칙, 히스토그램을 활용한 픽셀 연산 이해, 그리고 ROI를 학습 파이프라인과 연결하는 것입니다.[1][2][3]

## 한 줄 핵심
ROI는 이미지에서 관심 부분을 선택·가공·활용하는 **영역 선택** 기술이며, 좌표 체계와 슬라이싱 규칙을 정확히 이해하는 것이 중요합니다.[2][3]

## ROI 기본 개념
- ROI는 이미지에서 분석·가공·학습에 필요한 일부분을 의미합니다. 예를 들어 얼굴, 세포, 병변, 부품 같은 영역이 ROI입니다.[3][2]
- OpenCV 이미지는 넘파이 배열이므로, ROI는 단순 슬라이싱으로 얻을 수 있습니다. 이는 메모리 복사 없이 뷰(view)로 접근되는 경우가 많아 빠릅니다.[4][2]
- ROI는 일반적으로 사각형로 다루며, Mat/ndarray는 본질적으로 2D 그리드이기 때문에 비사각형은 마스크 등 별도 표현이 필요합니다.[5][4]

## 좌표와 슬라이싱
- OpenCV API는 좌표를 (x, y) 순서로 받는 경우가 많습니다. 반면 넘파이 슬라이싱은 [row, col] = [y, x] 순서입니다. 혼동을 피해야 합니다.[2][3]
- cv2.rectangle은 입력 이미지를 인플레이스(in-place)로 변경합니다. 원본 보존이 필요하면 .copy()를 사용하세요.[6][4]
- 슬라이싱은 시작 포함, 끝 제외 규칙입니다. 즉 [y1:y2, x1:x2]는 y1 ≤ y < y2, x1 ≤ x < x2 범위를 선택합니다.[7][2]

## 실전: ROI 추출 3가지
- 고정 좌표 슬라이싱: roi = img[y1:y2, x1:x2]. 단순·빠르고 파이프라인에 쉽습니다.[7][2]
- 인터랙티브 선택: cv2.selectROI로 드래그로 박스를 그려 선택. 엔터/스페이스로 확정, ESC 취소.[1]
- 컨투어/바운딩박스 기반: cv2.findContours → cv2.boundingRect → 슬라이싱으로 자동 ROI 추출.[8][9]

## 히스토그램과 선형 연산
- 히스토그램은 픽셀 분포를 보여줍니다. cv2.calcHist로 쉽게 계산합니다.[10][11]
- 밝기 선형 스케일링(예: ×1.7)을 ROI에 적용하면 분포가 우측으로 이동하고, 시각적으로 밝아집니다. 타입 변환과 클리핑에 유의하세요.[11][10]

## 코드: ROI 선택, 밝기 조정, 복원
- 목표: ROI 그리기 → 슬라이싱으로 ROI 추출 → 밝기 ×1.7 → 히스토그램 비교 → 원본 이미지에 ROI 덮어쓰기.[11][2]

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) 이미지 로드
img_color = cv2.imread('input.jpg', cv2.IMREAD_COLOR)  # BGR
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 2) 시각화용 사각형(인플레이스) — 원본 보존은 copy 사용
img_vis = img_color.copy()
# (x1, y1), (x2, y2)
x1, y1, x2, y2 = 753, 1210, 2115, 2110
cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255,255,255), 30)

# 3) ROI 슬라이싱은 [y, x] 순서
roi = img_gray[y1:y2, x1:x2]

# 4) 밝기 70% 상향: 클리핑과 dtype 주의
roi_up = np.clip(roi.astype(np.float32) * 1.7, 0, 255).astype(np.uint8)

# 5) 히스토그램 비교
hist_roi = cv2.calcHist([roi], , None, , [0,256])
hist_up  = cv2.calcHist([roi_up], , None, , [0,256])

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.plot(hist_roi); plt.xlim([0,256]); plt.title('ROI Hist')
plt.subplot(1,2,2); plt.plot(hist_up);  plt.xlim([0,256]); plt.title('ROI×1.7 Hist')
plt.show()

# 6) 원본에 덮어쓰기(인플레이스)
out = img_gray.copy()
out[y1:y2, x1:x2] = roi_up

plt.figure(figsize=(9,4))
plt.subplot(1,2,1); plt.imshow(img_gray, cmap='gray'); plt.axis('off'); plt.title('Gray')
plt.subplot(1,2,2); plt.imshow(out, cmap='gray'); plt.axis('off'); plt.title('Gray with ROI×1.7')
plt.show()
```


## 안전한 범위 처리와 좌표 유효성
- ROI가 이미지 경계를 넘지 않도록 np.clip으로 좌표를 보정하세요.[7][2]
- 음수 좌표나 x1≥x2, y1≥y2는 예외로 처리합니다. 인터랙티브 선택 시 (0,0,0,0)을 반환하면 건너뜁니다.[1][2]

```python
H, W = img_gray.shape[:2]
x1, y1 = np.clip([x1, y1], [0,0], [W, H])
x2, y2 = np.clip([x2, y2], [0,0], [W, H])
assert x2 > x1 and y2 > y1, "Invalid ROI"
```


## 고급: 마스크 기반 ROI와 주의
- 사각형 외 복잡한 ROI는 이진 마스크(0/1)로 표현하고, 연산은 img*mask + bg*(1-mask) 형태로 적용합니다.[12][5]
- 사각형 ROI에서 벗어난 경계 혼합이 필요하면 블렌딩이나 모폴로지 후처리를 고려합니다.[12][5]

## 연구 연결: ROI와 딥러닝
- ROI는 데이터 증강과 학습 안정화에 유용합니다. 자가 지도학습에서 의미 있는 오브젝트를 포함하는 오브젝트-어웨어 크롭은 표현학습에 도움을 줍니다.[13]
- ROI 마스크를 네트워크의 어텐션 맵으로 활용하여 관심 영역 기반 분류를 향상하는 접근도 연구되었습니다.[14]
- 의료 영상 등에서는 먼저 ROI를 지역화한 뒤 슬라이스별 세분화 성능을 개선하는 파이프라인이 실용적입니다.[15]

## 데이터 파이프라인 예시: ROI 중심 크롭 증강
- 목표: 데이터로더에서 입력마다 “오브젝트 포함 확률이 높은 ROI 크롭”을 생성합니다. 자가 지도나 분류/세분화 모두 적용 가능합니다.[13][2]

```python
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ROICropDataset(Dataset):
    def __init__(self, paths, roi_boxes=None, crop_size=224, p_obj=0.7):
        self.paths = paths
        self.roi_boxes = roi_boxes  # dict: path -> list of (x1,y1,x2,y2)
        self.crop_size = crop_size
        self.p_obj = p_obj

    def __len__(self):
        return len(self.paths)

    def _sample_crop(self, img, box=None):
        H, W = img.shape[:2]
        s = self.crop_size
        if box is not None and random.random() < self.p_obj:
            x1,y1,x2,y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            left = np.clip(cx - s//2, 0, max(0, W - s))
            top  = np.clip(cy - s//2, 0, max(0, H - s))
        else:
            left = random.randint(0, max(0, W - s))
            top  = random.randint(0, max(0, H - s))
        crop = img[top:top+s, left:left+s]
        return crop

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ROI가 여러 개인 경우 하나 샘플링
        box = None
        if self.roi_boxes and p in self.roi_boxes and len(self.roi_boxes[p]) > 0:
            box = random.choice(self.roi_boxes[p])

        crop = self._sample_crop(img, box)
        crop = cv2.resize(crop, (self.crop_size, self.crop_size))
        x = torch.from_numpy(crop).permute(2,0,1).float() / 255.0
        return x
```


## 모델 예시: ROI 조건 분류기
- ROI 마스크를 어텐션 맵으로 주입해 분류를 보조하는 간단한 브랜치 구조입니다. 마스크는 1채널로 리사이즈하여 이미지와 결합합니다.[14]

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ROIAttentionClassifier(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone  # e.g., torchvision.models.resnet18(pretrained=False)
        self.att_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),  # attention map in [0,1]
        )
        self.in_proj = nn.Conv2d(3, 3, 1)  # optional channel align

        # assume backbone.fc exists
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x, roi_mask=None):
        # x: (B,3,H,W), roi_mask: (B,1,H,W) or None
        if roi_mask is not None:
            a = self.att_conv(roi_mask)           # (B,1,H,W)
            x = x * (0.5 + 0.5*a)                 # amplify ROI softly
        x = self.in_proj(x)
        return self.backbone(x)
```


## 자가 지도 학습과 ROI
- 크롭 증강은 자가 지도에서 핵심입니다. 무작위 크롭 대신 오브젝트-어웨어 크롭을 섞으면 의미 있는 뷰의 상관성을 높일 수 있습니다.[13]
- 실무 팁: 라벨 없는 데이터에서 간단한 saliency/contour/threshold로 후보 ROI를 만들고, 상관 계수·에지 밀도 필터로 품질을 올립니다.[9][13]

## 검증: ROI 연산의 영향
- 밝기 스케일링은 히스토그램 이동으로 확인하세요. 분포가 우측으로 이동하면 밝기 증가가 반영된 것입니다.[10][11]
- 모델 입력의 분포 안정화를 위해, 학습·검증에 동일한 전처리(정규화, 크롭 정책)를 유지하세요.[3][2]

## 디버깅 체크리스트
- 좌표 순서 확인: OpenCV (x,y) vs 넘파이 [y,x].[3][2]
- 경계 초과 예외 처리: clip 및 폭/높이>0 보장.[2][1]
- dtype/클리핑: 연산 전 float32로 변환, 후 uint8 변환 및 np.clip 사용.[10][11]

## 추가 학습 자료
- OpenCV 공식 튜토리얼: ROI 크롭과 히스토그램 계산 요령을 정리해 둔 자료를 참고하면 좋습니다.[16][1]
- PyImageSearch: 크롭과 히스토그램 실전 예제가 자세합니다.[11][2]
- scikit-image: 마스크 기반 처리, 필터·형태학 연산 등 확장 기능이 풍부합니다.[12]

## 마무리
ROI는 단순 슬라이싱에서 시작해, 마스크·어텐션·오브젝트-어웨어 크롭으로 **학습 성능**을 끌어올리는 핵심 도구입니다. 좌표/슬라이싱/히스토그램의 기본기를 단단히 다지고, 데이터 파이프라인과 모델에 녹여 적용하면 연구와 실전에 모두 강력합니다.[13][2]

[1](https://opencv.org/blog/cropping-an-image-using-opencv/)
[2](https://pyimagesearch.com/2021/01/19/crop-image-with-opencv/)
[3](https://learnopencv.com/cropping-an-image-using-opencv/)
[4](https://docs.opencv.org/3.4/d3/d63/classcv_1_1Mat.html)
[5](https://answers.opencv.org/question/53781/roi-always-a-rectangle/)
[6](https://roboflow.com/use-opencv/draw-a-rectangle-with-cv2-rectangle)
[7](https://www.geeksforgeeks.org/python/crop-image-with-opencv-python/)
[8](https://stackoverflow.com/questions/15424852/region-of-interest-opencv-python)
[9](https://stackoverflow.com/questions/9084609/how-to-copy-a-image-region-using-opencv-in-python)
[10](https://www.geeksforgeeks.org/python/opencv-python-program-analyze-image-using-histogram/)
[11](https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/)
[12](https://arxiv.org/pdf/1407.6245.pdf)
[13](https://arxiv.org/pdf/2112.00319.pdf)
[14](https://arxiv.org/pdf/1812.00291.pdf)
[15](https://www.mdpi.com/2076-3417/11/4/1965/pdf?version=1614084313)
[16](https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html)
[17](https://toyprojects.tistory.com/10)
[18](http://science-gate.com/IJAAS/Articles/2022/2022-9-1/1021833ijaas202201016.pdf)
[19](https://www.frontiersin.org/articles/10.3389/fpls.2025.1511646/full)
[20](http://arxiv.org/pdf/1812.00155.pdf)
[21](https://www.mdpi.com/2072-4292/9/6/597/pdf?version=1497498870)
[22](https://www.mdpi.com/2306-5354/10/1/92/pdf?version=1673345930)
[23](https://www.mdpi.com/2076-3417/12/5/2674/pdf?version=1646392864)
[24](http://ijai.iaescore.com/index.php/IJAI/article/download/20493/12994)
[25](https://arxiv.org/pdf/1904.04441.pdf)
[26](https://arxiv.org/abs/2111.12309)
[27](https://arxiv.org/pdf/1704.02083.pdf)
[28](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-2-W16/61/2019/isprs-archives-XLII-2-W16-61-2019.pdf)
[29](https://www.frontiersin.org/articles/10.3389/fpls.2022.916474/pdf)
[30](https://www.techscience.com/JNM/v1n2/28978)
[31](https://www.mdpi.com/2072-4292/9/6/636/pdf?version=1498126018)
[32](https://arxiv.org/pdf/1809.01610.pdf)
[33](https://arxiv.org/pdf/2004.13665.pdf)
[34](https://www.youtube.com/watch?v=r-pp7flMoQA)
[35](https://wikidocs.net/208017)
[36](http://opencv-python.readthedocs.io/en/latest/doc/19.imageHistograms/imageHistograms.html)
[37](https://bkshin.tistory.com/entry/OpenCV-6-dd)
[38](https://www.reddit.com/r/opencv/comments/ew1syo/question_cv2rectangle_returning_back_the_same/)
[39](https://sikaleo.tistory.com/83)
[40](https://forum.opencv.org/t/resizing-scale-rectangle/6922)
[41](https://medipixel.github.io/post/2019-05-31-histogram/)

# Reference
- https://toyprojects.tistory.com/10
- image Region of Interest 좌표
