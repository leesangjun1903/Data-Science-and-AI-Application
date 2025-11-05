# An overview of gradient descent optimization algorithms

### 1. 핵심 주장과 주요 기여[1]

"An overview of gradient descent optimization algorithms"는 Sebastian Ruder가 2016년에 발표한 종합 개요 논문으로, 신경망 학습에서 널리 사용되는 다양한 경사하강법 알고리즘의 동작 원리와 장단점을 실무적 관점에서 설명합니다. 이 논문의 핵심 기여는 **검은 상자 최적화기로 다루어지던 알고리즘들에 대해 직관적인 이해를 제공**하는 것입니다.[1]

주요 주장은 다음과 같습니다:

- 미니배치 경사하강법이 현대 신경망 학습의 표준이 되었으나, 실무자들이 이들 알고리즘의 강점과 약점을 충분히 이해하지 못하고 있음
- 효과적인 모델 학습을 위해서는 알고리즘 선택이 데이터 특성에 따라 달라져야 함
- 모멘텀, 적응형 학습률, 분산 학습 등의 다양한 전략이 상이한 도전 과제를 해결

***

### 2. 문제 정의, 제안 방법, 모델 구조

#### 2.1 해결하고자 하는 문제들[1]

논문에서 다루는 주요 도전과제는:

1. **학습률 선택의 어려움**: 너무 작으면 수렴이 지연되고, 너무 크면 발산할 수 있음
2. **불균형한 데이터 특성**: 희소 데이터에서 매개변수마다 다른 업데이트 크기 필요
3. **안장점 문제(Saddle Point)**: 한 차원에서는 상향, 다른 차원에서는 하향하는 지점에서 경사도가 0에 가까워 탈출이 어려움
4. **메모리와 계산 효율성**: 대규모 데이터셋을 처리하는 계산량 문제

#### 2.2 제안하는 방법 및 수식

**a) 경사하강법 변형들**[1]

**배치 경사하강법 (Batch Gradient Descent)**:

$$
\theta = \theta - \eta \cdot \nabla_\theta J(\theta)
$$

전체 데이터셋에 대해 한 번에 업데이트하므로 안정적이나 느림.

**확률적 경사하강법 (Stochastic Gradient Descent)**:

$$
\theta = \theta - \eta \cdot \nabla_\theta J(\theta; x^{(i)}; y^{(i)})
$$

개별 샘플로 업데이트하여 빠르지만 변동성이 높음.

**미니배치 경사하강법 (Mini-batch Gradient Descent)**:

$$
\theta = \theta - \eta \cdot \nabla_\theta J(\theta; x^{(i:i+n)}; y^{(i:i+n)})
$$

배치와 SGD의 장점을 결합한 실무 표준.

**b) 가속화 알고리즘**[1]

**모멘텀 (Momentum)**:

$$
v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)
$$

$$
\theta = \theta - v_t
$$

과거 업데이트 방향을 누적하여 가속화. 협곡(ravine) 영역에서 진동을 감소시킴.

**Nesterov 가속 경사 (NAG)**:

$$
v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta - \gamma v_{t-1})
$$

$$
\theta = \theta - v_t
$$

미래 위치를 미리 살펴본 후 경사도를 계산하여 더 나은 반응성 제공.

**c) 적응형 학습률 알고리즘**[1]

**Adagrad**:

$$
\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii} + \epsilon}} \cdot g_{t,i}
$$

여기서 $$G_t$$는 과거 경사도 제곱의 합. 희소 데이터에 효과적이나 학습률이 단조감소.

**Adadelta**:

$$
\Delta\theta_t = -\frac{\text{RMS}[\Delta\theta]_{t-1}}{\text{RMS}[g]_t} g_t
$$

Adagrad의 단조감소 문제를 지수 이동평균으로 해결.

**RMSprop**:

$$
E[g^2]_t = 0.9E[g^2]_{t-1} + 0.1g^2_t
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$

Adadelta와 동일하게 지수 이동평균 사용하여 학습률 안정화.

**Adam (Adaptive Moment Estimation)**:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g^2_t
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta^t_1}, \quad \hat{v}_t = \frac{v_t}{1-\beta^t_2}
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t
$$

1차 모멘트(평균)와 2차 모멘트(분산)를 모두 활용하고 편향 보정 수행.

**AdaMax**:

$$
u_t = \max(\beta_2 \cdot v_{t-1}, |g_t|)
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{u_t}\hat{m}_t
$$

$$\ell_\infty$$ 노름을 사용하여 수치 안정성 개선.

**Nadam (Nesterov-accelerated Adam)**:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\left(\beta_1 \hat{m}_t + \frac{(1-\beta_1)g_t}{1-\beta^t_1}\right)
$$

Adam에 Nesterov 모멘텀을 통합하여 더 나은 응답성 제공.

---

### 3. 모델 일반화 성능 향상 관련 내용[1]

#### 3.1 배치 정규화 (Batch Normalization)

$$
\hat{x} = \frac{x - E[x]}{\sqrt{\text{Var}(x) + \epsilon}}
$$

배치 정규화는 각 미니배치 내에서 활성화를 정규화하여 다음의 이점을 제공합니다:[1]

- 더 높은 학습률 사용 가능
- 초기값에 대한 민감도 감소
- 정규화 효과로 인한 드롭아웃 필요성 감소
- 내부 공변량 이동(Internal Covariate Shift) 완화

#### 3.2 조기 종료 (Early Stopping)

검증 오류가 개선되지 않으면 학습을 중단하여 과적합 방지. 이는 모델의 **일반화 격차(generalization gap)** 감소에 효과적입니다.[1]

#### 3.3 데이터 셔플링 및 커리큘럼 학습

- **셔플링**: 학습 순서의 편향 제거
- **커리큘럼 학습**: 간단한 예제부터 어려운 예제로 점진적으로 학습시켜 더 나은 수렴성 달성[1]

#### 3.4 경사도 노이즈 추가

$$
g_{t,i} = g_{t,i} + N(0, \sigma^2_t)
$$

$$
\sigma^2_t = \frac{\eta}{(1+t)^\gamma}
$$

노이즈 추가는 모델이 더 깊은 신경망 학습에서 안장점 탈출을 도와 **일반화 성능 개선**에 기여합니다.[1]

***

### 4. 성능 향상과 한계

#### 4.1 알고리즘 비교[1]

논문의 시각화 분석(Figure 4)에서:

| 특성 | SGD | Momentum | NAG | Adagrad | RMSprop | Adadelta | Adam |
|------|-----|----------|-----|---------|---------|----------|------|
| 손실면 수렴성 | 느림 | 중간 | 빠름 | 빠름 | 빠름 | 매우 빠름 | 빠름 |
| 안장점 탈출 | 어려움 | 어려움 | 어려움 | 빠름 | 빠름 | 매우 빠름 | 빠름 |
| 기본값 설정 | 필요 | 필요 | 필요 | 불필요 | 불필요 | 불필요 | 불필요 |
| 희소 데이터 | 나쁨 | 나쁨 | 나쁨 | 우수 | 우수 | 우수 | 우수 |

#### 4.2 각 알고리즘의 한계

**Batch/SGD의 한계**:[1]
- 학습률이 고정되어 개별 매개변수 특성 반영 불가
- 안장점에서 탈출 어려움

**Adagrad의 한계**:[1]
- 학습률의 단조감소로 인해 후기 학습 정체
- 과거 모든 경사도 누적으로 메모리 사용

**Adam의 잠재적 문제**:
- 특정 상황에서 일반화 성능이 SGD-momentum에 비해 저하될 수 있음 (논문에서 직접 언급되진 않으나 이후 연구에서 지적)

***

### 5. 앞으로의 연구에 미치는 영향과 고려사항

#### 5.1 논문의 영향[2][3][1]

이 논문은 다음과 같은 방식으로 후속 연구에 영향을 미쳤습니다:

1. **표준화된 알고리즘 평가 체계 확립**: 논문에서 제시한 알고리즘 비교 방식이 기준점이 됨
2. **실무 적용의 확산**: Adam, RMSprop 등이 주요 프레임워크의 기본 최적화기로 채택
3. **최적화 연구의 방향 설정**: 적응형 학습률과 모멘텀의 조합이 주요 연구 주제로 발전

#### 5.2 최신 연구 동향 (2023-2025)[3][4][5][2]

**a) 아키텍처 인식 최적화 (Architecture-Aware Optimization)**[4][5]

최신 연구들은 신경망의 구조 정보를 활용한 최적화를 제안합니다:

- **자동 경사하강 (AGD)**: 네트워크 폭, 깊이 등 구조 특성에 맞춘 최적화기 자동 도출[4]
- **연결 인식 Adam (CaAdam)**: 계층별 연결 수와 깊이를 고려한 적응형 스케일링[5]
  - CIFAR-10에서 Adam 대비 4.09% 정확도 개선 (83.1% vs 79.8%)
  - CIFAR-100에서 2.87% RMSE 개선

**b) 동적 학습률 조정**[6][3]

- **RL 기반 학습률 적응**: Q-러닝을 활용한 동적 학습률 스케줄[6]
  - 전통적 스케줄 대비 더 빠른 수렴과 나은 최종 성능
  
- **적응형 학습률 스케줄러 (ASLR)**: 단일 하이퍼파라미터로 자동 조정[7]

**c) 일반화 격차 해결**[8][9][10][11]

- **대배치 학습의 일반화 격차**: 배치 크기 증가에 따른 일반화 성능 저하 문제 해결[11][8]
  - 학습 반복 횟수 조정과 Ghost Batch Normalization으로 5% → 1-2% 개선
  - 일반화 격차는 배치 크기가 아닌 **업데이트 횟수 부족**에서 비롯

- **불일치성과 불안정성**: 최신 이론적 분석에서 일반화 격차는 손실 지형의 날카로움보다 모델 출력의 **불일치성과 불안정성**으로 예측[12]

**d) 2차 최적화 방법의 재조명**[13][14][15]

- **Newton 방법, BFGS, 켤레 경사법**: 메모리와 계산 제약이 완화되면서 재평가 중[14][13]
- **Quiescence를 통한 2차 최적화**: 동적 시스템으로 최적화 궤적을 모델링[15]

**e) 다중 목표 최적화**[16]

- 여러 목표를 동시에 최적화하는 경사 기반 다중 목표 최적화 방법 개발[16]
- 멀티태스크 러닝, 다기준 러닝 등에 활용

#### 5.3 앞으로 연구 시 고려할 점

**1. 아키텍처 특성 활용**
- 모델의 너비, 깊이, 연결 구조 등을 최적화에 반영
- 일반화된 기본값보다는 구조 맞춤형 설정 추구

**2. 계산 효율성과 메모리 제약**
- GPU 메모리 최적화를 고려한 알고리즘 설계
- 대규모 모델(Transformers, LLMs)에 적합한 방법론 필요

**3. 이론과 실제의 격차 축소**
- 손실 지형의 기하학적 특성뿐 아니라 **출력의 일관성**을 최적화 기준으로 고려
- 오버파라미터화 신경망에서의 수렴성 보장

**4. 적응형 하이퍼파라미터 조정**
- 학습률 스케줄이 고정이 아닌 훈련 상태에 따른 동적 조정
- 강화학습, 메타러닝 기법 활용

**5. 모의 의료 이미지 처리 등 특정 도메인 최적화**
- 의료 영상(흉부 X선 뼈 억제 등) 학습에 최적화된 알고리즘 개발
- 데이터 불균형, 클래스 불균형 대응

**6. 정규화 기법 통합**
- 배치 정규화 외에 레이어 정규화, 그룹 정규화 등과 최적화기의 상호작용
- 조기 종료와 최적화기의 수렴성 관계 규명

#### 5.4 실무 적용 권장사항[17][1]

논문의 최종 권고사항과 최신 연구를 결합하면:

- **희소 데이터**: Adagrad 또는 Adam 계열 사용
- **일반적 신경망**: **Adam** 또는 **AdamW** (가중치 감쇠 분리) 추천
- **최종 성능 최대화**: SGD-momentum + 학습률 감쇠 스케줄 (더 나은 일반화 가능성)
- **빠른 수렴 필요**: Nadam 또는 아키텍처 인식 최적화 (CaAdam 등)
- **Transformer/대규모 모델**: 워밍업 + 코사인 감쇠 스케줄 + Adam 기반 최적화기 조합[18]

---

### 결론

Ruder의 논문은 2016년 당시 **경사하강법 최적화의 종합적 지형도**를 제시하여 신경망 학습의 실무적 이해를 크게 향상시켰습니다. 최근 연구들은 이 기초 위에서 **아키텍처 인식, 동적 조정, 이론적 기반** 세 가지 방향으로 진화 중입니다. 특히 의료 이미지 처리와 같은 특정 도메인에서는 이러한 발전된 최적화 기법들이 모델 성능과 일반화 능력을 크게 향상시킬 수 있으므로, 지속적인 모니터링과 실험적 검증이 필수실험적 검증이 필수적입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e55d6cec-40d5-4b97-8e4d-15faf709311a/1609.04747v2.pdf)
[2](https://arxiv.org/pdf/2302.09566v1.pdf)
[3](https://arxiv.org/abs/1909.13371v2)
[4](https://openreview.net/forum?id=DPZicYFdvD)
[5](https://arxiv.org/html/2410.24216v1)
[6](https://sakana.ai/assets/ai-scientist/rl_lr_adaptation.pdf)
[7](https://kastner.ucsd.edu/wp-content/uploads/2021/09/admin/ijcnn21-aslr.pdf)
[8](https://arxiv.org/pdf/1705.08741.pdf)
[9](http://arxiv.org/pdf/1906.01550.pdf)
[10](https://arxiv.org/pdf/2210.12184.pdf)
[11](https://papers.nips.cc/paper/6770-train-longer-generalize-better-closing-the-generalization-gap-in-large-batch-training-of-neural-networks)
[12](https://openreview.net/forum?id=5SIz31OGFV)
[13](https://www.linkedin.com/pulse/second-order-optimization-methods-abhijat-sarari-50xgc)
[14](https://www.geeksforgeeks.org/deep-learning/second-order-optimization-methods/)
[15](https://arxiv.org/abs/2410.08033)
[16](http://arxiv.org/pdf/2501.10945.pdf)
[17](https://www.ultralytics.com/glossary/adam-optimizer)
[18](https://eagle705.github.io/Learning-rate-warmup-scheduling/)
[19](https://arxiv.org/pdf/1903.03614.pdf)
[20](http://arxiv.org/pdf/2503.08489.pdf)
[21](https://arxiv.org/abs/2101.02397)
[22](https://arxiv.org/abs/2306.09778)
[23](http://arxiv.org/pdf/2309.06274.pdf)
[24](https://optimization-online.org/2024/11/some-new-accelerated-and-stochastic-gradient-descent-algorithms-based-on-locally-lipschitz-gradient-constants/)
[25](https://www.sciltp.com/journals/ijndi/article/view/522)
[26](https://www.nature.com/articles/s41598-025-20788-y)
[27](https://arxiv.org/pdf/1510.04609.pdf)
[28](https://arxiv.org/pdf/2409.04707.pdf)
[29](https://arxiv.org/pdf/2112.03660.pdf)
[30](http://arxiv.org/pdf/2503.18219.pdf)
[31](http://arxiv.org/pdf/2306.00169.pdf)
[32](http://arxiv.org/pdf/2201.11022.pdf)
[33](https://arxiv.org/pdf/2207.06080.pdf)
[34](https://www.mathworks.com/help/deeplearning/ref/warmuplearnrate.html)
[35](http://papers.neurips.cc/paper/6770-train-longer-generalize-better-closing-the-generalization-gap-in-large-batch-training-of-neural-networks.pdf)
[36](https://discuss.pytorch.org/t/using-both-learning-rate-warm-up-and-a-learning-rate-scheduler/177767)
[37](https://arxiv.org/abs/2210.12184)
