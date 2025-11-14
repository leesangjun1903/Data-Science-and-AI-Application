
# Enhanced Transport Distance for Unsupervised Domain Adaptation

## 1. 핵심 주장과 주요 기여 요약

**Enhanced Transport Distance (ETD)**는 최적 수송(Optimal Transport, OT) 이론을 기반으로 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 문제를 해결하는 방법이다. 이 논문의 핵심 주장은 기존 OT 기반 도메인 적응 방법들이 **판별 정보(discriminant information)** 와 **카테고리 사전 지식(category prior)** 을 충분히 활용하지 못한다는 점에 있다.[1]

### 주요 기여

1. **주의 기반 거리 가중화(Attention-based Distance Weighing)**: 분류기의 예측 피드백을 활용하여 동적으로 샘플 간 유사성을 재평가하고, 이를 통해 OT 거리를 가중화한다.[1]

2. **칸토로비치 퍼텐셜의 신경망 재매개변수화(Network Re-parameterization of Kantorovich Potential)**: 전통적인 벡터 형태의 이중 변수 대신 심층 신경망(3층 완전연결층)을 사용하여 더 정확한 OT 계획을 학습한다.[1]

3. **엔트로피 정규화**: 타겟 도메인의 내재 구조를 탐색하기 위해 엔트로피 기준을 목적 함수에 추가한다.[1]

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 문제 정의

UDA는 충분한 라벨을 가진 소스 도메인 데이터와 라벨이 없는 타겟 도메인 데이터를 활용하여 모델의 분류 성능을 향상시키는 문제이다. 핵심 과제는 **도메인 시프트(domain shift)** 로 인한 성능 저하를 최소화하는 것이다.[1]

기존 OT 기반 방법들의 한계:[1]

- **미니배치 학습의 편향**: 미니배치 내 샘플들은 실제 분포를 완전히 대표하지 못하여 추정된 OT 계획이 편향된다.
- **판별 정보 부족**: 타겟 도메인의 라벨 정보와 구조를 무시한다.

### 2.2 제안 방법 및 수식

#### 2.2.1 주의 기반 거리 가중화

분류기의 출력 $$η(f^s), η(f^t)$$ 를 활용하여 주의 행렬 $$S \in \mathbb{R}^{b \times b}$$ 를 계산한다:[1]

$$
S_{ij} = \sigma\left(\mathbf{W}_a^s η(f_i^s)^T \mathbf{W}_a^t η(f_j^t)\right)
$$

여기서 $$\sigma(\cdot)$$ 는 활성화 함수(ReLU 사용)이고, $$\mathbf{W}_a^s, \mathbf{W}_a^t$$ 는 투영 행렬이다. 정규화된 주의 행렬을 이용하여 거리 행렬을 재가중화한다:[1]

$$
\tilde{C} = S \odot C
$$

여기서 $$\odot$$ 는 요소별 곱셈(Hadamard product)이고, $$C_{ij} = c(f_i^s, f_j^t)$$ 이다.[1]

#### 2.2.2 최적 수송의 반이중(Semi-dual) 공식

정규화된 OT 문제의 반이중 공식은:[1]

$$
\sup_v \mathbb{E}_{f^s \sim \mu}\left[v^{c,\varepsilon}(f^s)\right] + \mathbb{E}_{f^t \sim \nu}\left[v(f^t)\right] - \varepsilon
$$

여기서 $$v$$ 는 칸토로비치 퍼텐셜이고:

$$
v^{c,\varepsilon}(f^s) = -\varepsilon \log \mathbb{E}_{f^t \sim \nu}\left[\exp\left(\frac{v(f^t) - c(f^s, f^t)}{\varepsilon}\right)\right], \quad \varepsilon > 0
$$

#### 2.2.3 신경망 기반 칸토로비치 퍼텐셜

전통적인 벡터 형태의 이중 변수 대신 신경망 $$g(·)$$ 로 재매개변수화하여 최적화 문제를 다음과 같이 변환한다:[1]

$$
\max_g H_\varepsilon(g(f^t)) = \sum_{i=1}^b \left[\sum_{j=1}^b g(f_j^t) + h_\varepsilon(f_i^s, g(f^t))\right]
$$

여기서:

$$
h_\varepsilon(f_i^s, g(f^t)) = \begin{cases}
\varepsilon \log \sum_{j=1}^b \exp\left(\frac{g(f_j^t) - c(f_i^s, f_j^t)}{\varepsilon}\right) - \varepsilon, & \varepsilon > 0 \\
\min_j \left(c(f_i^s, f_j^t) - g(f_j^t)\right), & \varepsilon = 0
\end{cases}
$$

최종 OT 거리는:[1]

$$
W_\varepsilon(\mu, \nu) = \mathbb{E}_{f^s \sim \mu}\left[g^{c,\varepsilon}(f^s)\right] + \mathbb{E}_{f^t \sim \nu}\left[g(f^t)\right] - \varepsilon
$$

#### 2.2.4 전체 목적 함수

전체 손실 함수는 세 가지 항으로 구성된다:[1]

$$
L(\mathbf{W}) = L_s(\mathbf{W}) + \lambda L_{opt}(\mathbf{W}) + \beta L_t(\mathbf{W})
$$

여기서:

- **소스 분류 손실**: $$L_s(\mathbf{W}) = \frac{1}{n_s}\sum_{i=1}^{n_s} l_{ce}(η(f_i^s), y_i^s; \mathbf{W})$$
- **도메인 적응 손실**: $$L_{opt}(\mathbf{W}) = W_\varepsilon(\mu, \nu)$$
- **타겟 엔트로피 손실**: $$L_t(\mathbf{W}) = \frac{1}{n_t}\sum_{i=1}^{n_t}\sum_{j=1}^C -\hat{y}\_{ij}^t \log \hat{y}_{ij}^t$$

### 2.3 모델 구조

모델은 세 개의 주요 구성 요소로 이루어져 있다:[1]

1. **특징 추출 네트워크** $$f(·)$$: 입력 이미지에서 심층 특징을 추출하며, 소스와 타겟 도메인이 가중치를 공유한다.

2. **분류 네트워크** $$η(·)$$: 추출된 특징을 분류 예측으로 변환한다. 주의 메커니즘의 입력으로 사용된다.

3. **칸토로비치 퍼텐셜 네트워크** $$g(·)$$: 세 개의 완전연결층으로 구성되어 있으며, OT 계획을 학습한다.

### 2.4 성능 향상

실험 결과는 다음과 같다:[1]

| 데이터셋 | 평균 정확도 | 이전 SOTA 대비 개선 |
|---------|-----------|-----------------|
| Office-31 | 86.2% | 기본 수준 유지 |
| ImageCLEF-DA | 89.7% | +2% (CDAN+E 대비) |
| Office-Home | 67.3% | +1.5% (CDAN 대비) |
| Digits | 96.9% | +2.7% (평균) |

주요 성과:[1]

- **Office-Home**: 특히 Ar→Rw 작업에서 9.7% 정확도 향상
- **ImageCLEF-DA**: I→P 작업에서 3.3% 개선
- **Digits Recognition**: S→M 작업에서 최고 정확도 달성

### 2.5 한계 및 제약

1. **하이퍼파라미터 민감도**: λ와 ε 파라미터가 데이터셋에 따라 서로 다른 최적값을 가진다. 예를 들어, λ는 Office-Home에서 0.5, Digits에서 25로 크게 차이난다.[1]

2. **계산 복잡도**: 신경망 기반 퍼텐셜 최적화로 인한 추가 계산 오버헤드가 발생한다.[1]

3. **이중 루프 최적화**: 알고리즘이 두 개의 중첩된 루프(내부 OT 최적화, 외부 네트워크 파라미터 업데이트)를 사용하여 수렴이 느릴 수 있다.[1]

4. **제한된 이론적 보장**: 신경망 기반 재매개변수화의 수렴성에 대한 엄격한 이론적 분석이 부족하다.[1]

---

## 3. 일반화 성능 향상 가능성 분석

### 3.1 특징 공간 변환의 효과

t-SNE 시각화 결과에서 적응 후 명확한 클러스터 구조가 형성된다. 이는 다음을 의미한다:[1]

- **클러스터 중심 근접화**: 소스와 타겟 도메인의 클래스 중심이 서로 가까워진다.
- **분산도 유사성**: 클러스터의 퍼짐 정도가 비슷해져 균형 잡힌 분포가 형성된다.

### 3.2 주의 메커니즘의 역할

주의 기반 거리 가중화는 다음과 같이 일반화 성능을 향상시킨다:[1]

1. **샘플별 중요도 재조정**: 분류기의 신뢰도가 높은 샘플 쌍에 더 높은 가중치를 부여한다.
2. **판별 정보 활용**: 분류 예측을 직접 활용하여 도메인 적응 과정에 피드백한다.
3. **미니배치 편향 완화**: 현재 배치의 실제 특성을 동적으로 반영한다.

### 3.3 엔트로피 정규화의 기여

타겟 도메인 엔트로피 손실 $$L_t$$ 는:[1]

- **클래스 구조 탐색**: 타겟 데이터의 내재 분포 구조를 활용한다.
- **의사 라벨 품질 개선**: 분류기의 신뢰도 높은 예측을 강화한다.
- **부정적 전이 감소**: 신뢰도 낮은 예측을 억제하여 잘못된 적응을 방지한다.

### 3.4 일반화 성능의 제약

1. **도메인 갭의 크기**: Office-Home과 같이 큰 도메인 갭이 있는 경우 성능 향상이 제한적이다.[1]
2. **미니배치 크기 의존성**: 알고리즘이 미니배치 기반이므로, 배치 크기가 작을 경우 불안정할 수 있다.[1]
3. **카테고리 수 영향**: 카테고리가 많을수록 (Office-Home 65개 vs Digits 10개) 적응이 어려워진다.[1]

***

## 4. 연구 영향 및 향후 연구 고려사항

### 4.1 학술적 영향

#### 최적 수송의 새로운 활용

ETD는 다음과 같은 측면에서 최적 수송 이론의 적용 범위를 확장한다:[2]

1. **신경망 기반 매개변수화**: 이중 변수를 신경망으로 재매개변수화하는 아이디어는 다른 OT 기반 응용에도 적용될 수 있다.
2. **주의 메커니즘과 OT의 결합**: 판별 정보를 기반으로 OT 계획을 가중화하는 새로운 패러다임을 제시한다.

#### 도메인 적응 연구의 방향성

최근 연구 경향을 보면:[3][4]

1. **기초 모델의 활용**: 최신 연구는 CLIP 같은 시각-언어 모델의 일반화 능력을 활용하려 한다.[5]
2. **글로벌-로컬 정렬**: 단순 전역 특징 정렬을 넘어 카테고리 수준과 픽셀 수준의 다중 그래뉼러리티 정렬이 중요해지고 있다.[6]

### 4.2 현재 도메인 적응 연구의 주요 과제

#### 4.2.1 오픈셋 도메인 적응 (Open Set Domain Adaptation, OSDA)

현실 세계 환경에서는 타겟 도메인에 소스 도메인에 없는 알려지지 않은 클래스가 존재한다. 이는 ETD 같은 방법들이 모든 타겟 샘플을 정렬하려 하기 때문에 **부정적 전이(negative transfer)** 를 야기할 수 있다. ETD는 이 문제를 명시적으로 다루지 않으므로, 향후 오픈셋 시나리오에 적응해야 한다.[7]

#### 4.2.2 소스 프리 도메인 적응 (Source-Free Domain Adaptation, SFDA)

최근 연구는 소스 데이터에 접근할 수 없는 상황에 집중하고 있다. 이는 다음과 같은 실용적 이유가 있다:[8]

- 지적 재산권 보호
- 개인정보 보안
- 계산 리소스 제약

ETD는 학습 중 소스 데이터를 필요로 하므로, 소스 프리 설정으로의 확장이 필요하다.[9]

#### 4.2.3 지속적 도메인 시프트 (Continual Domain Shift)

실제 배포 시나리오에서는 새로운 도메인이 시간에 따라 계속해서 나타난다. 이는 다음의 도전 과제를 야기한다:[10][11]

- 재앙적 망각(catastrophic forgetting)
- 누적 성능 저하
- 적응적 수렴 조건의 결정

#### 4.2.4 부분 도메인 적응 (Partial Domain Adaptation)

타겟 도메인이 소스 도메인의 부분 집합인 경우를 다루는 문제이다. 이 경우, 불필요한 클래스의 샘플을 강제로 정렬하는 것은 해로울 수 있다.[12]

### 4.3 향후 연구 시 고려할 점

#### 4.3.1 스케일 및 효율성

현재 최신 연구 방향:[13]

- **글로벌 인식 강화**: 미니배치를 넘어 전체 데이터 분포의 통계적·기하학적 특성을 고려한 방법들이 개발되고 있다.
- **계산 효율성**: OT의 계산 복잡도 O(n³ log n)을 줄이기 위한 근사 방법들이 활발히 연구 중이다.[14]

#### 4.3.2 기초 모델과의 통합

최신 시각 트랜스포머 기반 접근:[15]

- 사전 학습된 기초 모델은 이미 강력한 일반화 능력을 가지고 있다.
- ETD의 아이디어를 트랜스포머 기반 모델에 적용하면 더 강력한 결과를 얻을 수 있을 것으로 예상된다.

#### 4.3.3 다중 소스 도메인 적응

일반화를 더욱 강화하기 위해:[16]

- 여러 소스 도메인으로부터 강건한 표현을 학습해야 한다.
- 분포 이동 환경에서의 강건성을 고려한 적응 전략이 필요하다.

#### 4.3.4 이론적 보증 강화

현재 ETD의 수렴성 보증이 제한적이므로:[17]

- 신경망 기반 OT 재매개변수화의 수렴 조건에 대한 엄격한 분석
- 주의 기반 가중화의 최적성 보증
- 미니배치 설정 하에서의 편향-분산 분석

#### 4.3.5 실제 응용 시나리오 대응

의료 영상 같은 고위험 분야에서:[18]

- 스캐너 도메인 시프트 같은 특정 도메인 시프트에 대한 강건성
- 분포 시프트 유형의 정확한 진단 및 대응
- 불확실성 정량화

### 4.4 ETD의 미래 연구 방향

#### 단기 (1-2년)

1. **오픈셋 시나리오 지원**: ETD를 부분 OT나 선택적 OT로 확장하여 미지의 클래스 처리
2. **소스 프리 변형**: 사전 학습된 모델 기반의 적응 가능한 변형 개발
3. **기초 모델 통합**: ViT나 CLIP 같은 기초 모델과의 결합 연구

#### 중기 (2-5년)

1. **지속적 학습 통합**: 시간에 따른 새로운 도메인 도착에 대응하는 방법론
2. **다중 소스 최적화**: 여러 소스로부터의 가중 결합 전략
3. **이론-실무 간극 축소**: 더욱 강력한 수렴 보증 및 성능 경계

#### 장기 (5년 이상)

1. **통합 전이 학습 프레임워크**: 도메인 적응, 도메인 일반화, OOD 탐지를 통합
2. **자체 적응 메커니즘**: 하이퍼파라미터 자동 조정
3. **멀티모달 도메인 적응**: 텍스트, 이미지, 시계열 등 다양한 데이터 타입 지원

---

## 결론

**Enhanced Transport Distance (ETD)** 는 최적 수송 이론에 주의 메커니즘과 신경망 기반 재매개변수화를 결합하여 비지도 도메인 적응 성능을 향상시킨 중요한 기여이다. 특히 주의 기반 거리 가중화와 엔트로피 정규화는 판별 정보를 효과적으로 활용하는 새로운 패러다임을 제시한다.[1]

그러나 현재 도메인 적응 연구는 오픈셋 도메인 적응, 소스 프리 적응, 지속적 도메인 시프트, 기초 모델 통합 등 새로운 과제들에 직면해 있다. ETD의 핵심 아이디어들은 이러한 미래 문제들에 대응하기 위한 기반을 제공할 것으로 기대되며, 신경망 기반 OT 재매개변수화는 더 나은 계산 효율성과 표현 능력을 갖춘 다음 세대 도메인 적응 방법들의 토대가 될 수 있을 것이다.[4][3][13][6][8]

---

## 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/547c03e0-3681-4ef1-969c-0964cbea29b0/Li_Enhanced_Transport_Distance_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf)
[2](https://arxiv.org/pdf/2208.07422.pdf)
[3](https://www.mdpi.com/1099-4300/27/4/426)
[4](https://arxiv.org/pdf/2210.03885.pdf)
[5](https://www.ijcai.org/proceedings/2025/0165.pdf)
[6](https://www.tandfonline.com/doi/full/10.1080/01431161.2025.2450564?ai=179&mi=l49ppp&af=R)
[7](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Separate_to_Adapt_Open_Set_Domain_Adaptation_via_Progressive_Separation_CVPR_2019_paper.pdf)
[8](https://www.sciencedirect.com/science/article/abs/pii/S0925231223010445)
[9](http://arxiv.org/pdf/2212.04227.pdf)
[10](https://arxiv.org/pdf/2303.15833.pdf)
[11](http://arxiv.org/pdf/2402.00580.pdf)
[12](https://www.ijcai.org/proceedings/2020/0352.pdf)
[13](https://arxiv.org/html/2502.06272v1)
[14](https://bridges.monash.edu/articles/thesis/Optimal_Transport_Theory_for_Domain_Adaptation/25016954)
[15](https://openreview.net/forum?id=ATdshE4yIj)
[16](https://arxiv.org/pdf/2309.02211.pdf)
[17](http://arxiv.org/pdf/2502.19316.pdf)
[18](https://arxiv.org/abs/2409.04368)
[19](https://arxiv.org/pdf/2201.01806.pdf)
[20](https://arxiv.org/pdf/2110.12024.pdf)
[21](https://arxiv.org/html/2411.15844)
[22](https://arxiv.org/pdf/2501.16410.pdf)
[23](http://www.conf-icnc.org/2024/papers/p506-ahamed.pdf)
[24](https://proceedings.mlr.press/v119/chuang20a/chuang20a.pdf)
[25](https://arxiv.org/abs/2208.07422)
[26](https://www.sciencedirect.com/science/article/abs/pii/S0957417425024686)
[27](https://arxiv.org/html/2510.03540v1)
[28](https://arxiv.org/html/2503.13868v3)
[29](https://www.ijcai.org/proceedings/2024/0127.pdf)
[30](https://cvpr.thecvf.com/virtual/2025/poster/32754)
[31](https://arxiv.org/html/2411.12558v1)
