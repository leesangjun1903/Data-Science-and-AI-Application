# Cubic Convolution Interpolation 이해하기

# 요약  
이미지 처리와 딥러닝 모델 구축에 필수적인 **Cubic Convolution Interpolation**(삼차 합성 곡선 보간법)의 원리부터 구현 예시까지 빠르게 훑어보고, 실제 코드로 직접 실습해봅니다.

# 1. Cubic Convolution Interpolation 개념  
Cubic Convolution Interpolation은 픽셀 값을 1차, 2차가 아닌 **3차 다항식**으로 모델링하여, 입력 영상보다 더 부드러운 중간 값을 계산하는 기법입니다.  
- **선형 보간(linear interpolation)**: 두 점을 직선으로 연결  
- **2차 보간(quadratic interpolation)**: 세 점을 이용한 곡선  
- **3차 보간(cubic convolution)**: 네 점을 활용한 다항식  
네 점을 활용해 주변 픽셀 p(−1), p(0), p(1), p(2)에 곡선을 맞추고, 새로운 좌표 x에 대한 픽셀 값을 계산합니다.

# 2. 수식 해설  
삼차 합성 곡선 보간의 핵심 수식은 다음과 같습니다.  

$$  
I(x) = \sum_{k=-1}^{2} p_k \, h(x - k)  
$$  

여기서 h(t)는 **커널 함수**로, 보통 다음과 같은 형태를 가집니다.  

$$  
h(t) =  
\begin{cases}  
( a + 2)|t|^3 - ( a + 3)|t|^2 + 1, & |t| < 1 \\  
a|t|^3 - 5a|t|^2 + 8a|t| - 4a, & 1 \le |t| < 2 \\  
0, & \text{otherwise}  
\end{cases}  
$$  

- 여기서 **a**는 곡선의 형태를 조절하는 파라미터(일반적으로 −0.5 또는 −0.75 사용).  
- |t|<1 구간은 중앙 네 점에 대한 가중치를, 1≤|t|<2 구간은 외곽 점 기여도를 조정합니다.

# 3. 파이썬 구현 예제  
짧은 코드로 1D 신호에 대한 Cubic Convolution Interpolation을 구현해보겠습니다.  

```python
import numpy as np

def cubic_kernel(t, a=-0.5):
    t = abs(t)
    if t < 1:
        return (a+2)*t**3 - (a+3)*t**2 + 1
    elif t < 2:
        return a*t**3 - 5*a*t**2 + 8*a*t - 4*a
    else:
        return 0

def cubic_interp1d(signal, x_new, a=-0.5):
    output = []
    for x in x_new:
        i = int(np.floor(x))
        val = 0.0
        for k in range(-1, 3):
            idx = min(max(i+k, 0), len(signal)-1)
            val += signal[idx] * cubic_kernel(x - (i+k), a)
        output.append(val)
    return np.array(output)

# 예시 데이터
orig = np.linspace(0, 10, 5)             # [0, 2.5, 5, 7.5, 10]
xq = np.linspace(0, 4, 17)               # 더 촘촘한 좌표
interp = cubic_interp1d(orig, xq)

print(interp)
```

- `cubic_kernel` 함수: 커널 수식을 구현  
- `cubic_interp1d`: 원본 신호와 새로운 좌표(x_new)로 보간 결과 반환  

# 4. 이미지에 적용하기  
1D 예제를 2D 이미지로 확대하면 각 축(x, y)에 대해 1D 보간을 두 번 적용합니다.  
```python
import cv2

def resize_cubic(img, new_h, new_w, a=-0.5):
    h, w = img.shape[:2]
    # x축 보간
    x_coords = np.linspace(0, w-1, new_w)
    tmp = np.zeros((h, new_w), dtype=np.float32)
    for i in range(h):
        tmp[i, :] = cubic_interp1d(img[i, :], x_coords, a)
    # y축 보간
    y_coords = np.linspace(0, h-1, new_h)
    out = np.zeros((new_h, new_w), dtype=np.float32)
    for j in range(new_w):
        out[:, j] = cubic_interp1d(tmp[:, j], y_coords, a)
    return out.astype(img.dtype)

# 실제 사용
img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
resized = resize_cubic(img, 512, 512)
cv2.imwrite('resized.jpg', resized)
```

# 5. 딥러닝 모델 내 활용  
딥러닝 프레임워크(TensorFlow, PyTorch)에서는 기본으로 제공하는 **`interpolate`**, **`upsample`** 함수에서 커널 옵션으로 `cubic`을 선택하면 내부적으로 유사 동작을 수행합니다.  
```python
import torch
import torch.nn.functional as F

x = torch.randn(1, 3, 128, 128)
upsampled = F.interpolate(x, size=(256,256), mode='bicubic', align_corners=False)
```

# 결론  
Cubic Convolution Interpolation은 단순 업샘플링보다 부드러운 결과를 얻을 수 있어, 이미지 전처리나 Super-Resolution 같은 딥러닝 분야에서 널리 사용됩니다. 위 예제를 바탕으로 직접 구현해보고, 프레임워크 내장 함수를 비교 실험해보세요!
