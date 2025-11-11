# Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

### 1. 핵심 주장과 주요 기여 (간결 요약)[1]

본 논문은 **정류 활성화 함수(Rectified Activation Units)**의 특성을 체계적으로 연구하여 ImageNet 분류에서 인간 수준의 성능을 최초로 초과했습니다. 주요 기여는 두 가지입니다:[1]

**Parametric Rectified Linear Unit (PReLU)**: ReLU의 음수 부분을 학습 가능한 매개변수 $$a_i$$로 일반화하여, 무시할 수 있는 계산 비용으로 1.05~1.2% 정확도를 개선했습니다.[1]

**He 초기화**: ReLU/PReLU의 비선형성을 명시적으로 고려한 초기화 이론으로, 30층까지의 매우 깊은 네트워크를 처음부터 직접 학습할 수 있게 했으며, 최종적으로 4.94% top-5 오류율로 인간 성능(5.1%)을 초과했습니다.[1]

***

### 2. 해결 문제, 제안 방법, 모델 구조

#### 2.1 해결하는 문제[1]

ReLU의 광범위한 사용에도 불구하고 초기화 전략이 미흡했으며, 매우 깊은 네트워크(30층 이상)에서 기울기 소실 문제가 발생했고, Xavier 초기화가 ReLU의 비대칭성을 고려하지 못했습니다.[1]

#### 2.2 PReLU 활성화 함수[1]

**정의**:

$$
f(y_i) = \begin{cases}
y_i, & \text{if } y_i > 0 \\
a_i y_i, & \text{if } y_i \leq 0
\end{cases}
$$

여기서 $$a_i$$는 i번째 채널의 음수 부분 기울기를 제어하는 학습 가능 계수입니다. 역전파 기울도는 다음과 같습니다:[1]

$$
\frac{\partial E}{\partial a_i} = \sum_{y_i} \frac{\partial E}{\partial f(y_i)} \cdot y_i \quad (\text{when } y_i \leq 0)
$$

모멘텀 업데이트는 가중치 감소를 적용하지 않습니다:[1]

$$
\Delta a_i := \mu \Delta a_i + \epsilon \frac{\partial E}{\partial a_i}
$$

#### 2.3 He 초기화 이론[1]

**정방향 전파 분석**: ReLU 활성화에서 기울기 $$E[x_l^2] \neq \text{Var}[x_l]$$이므로:

$$
\text{Var}[y_l] = \frac{1}{2} n_l \text{Var}[w_l] \text{Var}[y_{l-1}]
$$

L층 누적 분산을 안정화하려면:

$$
\frac{1}{2} n_l \text{Var}[w_l] = 1 \quad \Rightarrow \quad \text{std} = \sqrt{\frac{2}{n_l}}
$$

**역방향 전파**: 동일하게:

$$
\frac{1}{2} \hat{n}_l \text{Var}[w_l] = 1 \quad \Rightarrow \quad \text{std} = \sqrt{\frac{2}{\hat{n}_l}}
$$

**Xavier 초기화와 비교**: Xavier는 $$\sqrt{1/n_l}$$의 표준편차를 사용하므로, He 초기화는 정규화 계수 2를 추가로 적용하여 ReLU의 비대칭성을 반영합니다.[1]

#### 2.4 모델 아키텍처[1]

| 모델 | 깊이 | 특징 |
|------|------|------|
| **Model A** | 19층 | 첫 층 7×7 stride 2, SPP (7×7, 3×3, 2×2, 1×1) |
| **Model B** | 22층 | Model A + 3개 conv층 |
| **Model C** | 22층 | Model B의 필터 수 2배 증가 (권장 설정) |

주요 학습 설정:[1]
- 초기화: He 초기화 (표준편차 $$\sqrt{2/n}$$)
- 활성화: PReLU (채널별)
- 데이터 증강: 전체 훈련 과정에 스케일 지터링(256-512), 임의 자르기, 색상 변동 적용
- 배치 크기: 128, 학습률: 1e-2 → 1e-3 → 1e-4

***

### 3. 성능 향상 및 일반화

#### 3.1 정성 성능 비교[1]

**PReLU 효과 (14층 모델)**:

| 모델 | Top-1 | Top-5 |
|------|-------|-------|
| ReLU | 33.82% | 13.34% |
| PReLU (채널별) | 32.64% | 12.75% |
| **개선** | **1.18%** | **0.59%** |

#### 3.2 최종 결과 - 인간 성능 초과[1]

**테스트 세트 다중 모델 결과**:

| 팀 | Top-5 오류 |
|----|---------|
| GoogLeNet (2014) | 6.66% |
| Baidu | 5.98% |
| **MSRA (PReLU-nets)** | **4.94%** |
| **인간 성능** | **5.1%** |

**의의**: 26% 상대적 개선, **최초로 인간 수준 성능 초과**[1]

#### 3.3 일반화 성능 중점 분석[1]

**깊이와 폭의 트레이드오프**: 논문 내에서 30층 모델이 14층 모델보다 정확도가 악화되는 현상을 관찰했으며, 이를 해결하기 위해 깊이 증가 대신 **폭(필터 수) 증가를 선택**했습니다.[1]

**기울기 흐름 안정성**: He 초기화는 신호의 분산을 정방향과 역방향 모두에서 보존하여, 매우 깊은 네트워크에서도 기울기 소실을 방지합니다.[1]

**데이터 증강**: 적극적 데이터 증강(스케일 지터링, 색상 변동)으로 큰 모델의 과적합을 방지하고 일반화를 개선했습니다.[1]

---

### 4. 모델 한계[1]

1. **깊이 증가의 한계**: 30층 이상에서 정확도 악화 - 단순히 깊이만으로는 성능 개선 불가
2. **특정 작업의 어려움**: 문맥 이해나 고수준 지식 필요한 작업(예: 식당 인식) 미흡
3. **계산 비용**: Model C 학습에 4개 K40 GPU로 3-4주 소요
4. **앙상블 의존성**: 최종 결과는 6개 모델 앙상블 기반

***

### 5. 향후 연구에 미친 영향 및 향후 고려사항

#### 5.1 핵심 기여의 영속적 영향[2][3][4]

**He 초기화의 광범위 채택**: 현재(2025년) 거의 모든 ReLU 기반 심층 신경망의 표준 초기화로 사용되며, Vision Transformer, ResNet, CNN 등 폭넓은 아키텍처에 적용됩니다.[3][4][2]

**학습 가능 활성화 발전**: PReLU 개념은 ELU(2016), Swish, GELU 등 학습/적응 활성화 함수 개발을 촉발했습니다.[5][6]

**깊이 vs 폭 문제 해결**: 본 논문의 한계는 ResNet(2015)의 스킵 연결로 해결되었으며, 이후 깊이와 폭의 이론적 트레이드오프가 증명되었습니다.[7][8]

#### 5.2 의료 영상 분석 분야 영향[9]

He 초기화와 PReLU는 **전이 학습 기반 의료 영상 분류**의 기반이 되었으며, 제한된 의료 데이터로도 깊은 네트워크의 안정적 학습을 가능하게 했습니다. 특히 다중 작업 학습에서 더 나은 일반화를 달성했습니다.[9]

#### 5.3 향후 연구 시 고려할 점 (최신 연구 기반)

**활성화 함수 설계**: 도메인/작업별 최적 활성화 함수는 다르며, 메타 학습이나 자동 아키텍처 탐색(NAS)을 통한 자동 최적화 필요[10][11][5]

**초기화 방법 개선**: 현재 He 초기화의 이론-실제 갭(특정 층에 수동 조정 필요)을 보완하기 위해 적응적/도메인 특화 초기화 개발 진행 중[12][13]

**깊이 vs 폭 최적화**: 신경 아키텍처 탐색(NAS)과 프루닝을 통한 자동 최적화, 다이나믹 깊이(입력 복잡도에 따른 동적 조정) 연구 진행[14][7]

**배치 정규화와의 상호작용**: 최신 연구에서 배치 정규화 없는 훈련에서도 He 초기화의 중요성이 재검증되었으며, 엣지 디바이스 배포 시 가중치를 받고 있습니다.[15][16]

**일반화 이론 심화**: 정보 이론적 분석(2024)으로 깊이의 일반화 효과를 정량화하고, 일반화 한계 개선 연구가 계속 진행 중입니다.[17][18]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2702b4c1-1c03-4774-b07e-76ee0e37fc03/1502.01852v1.pdf)
[2](https://wandb.ai/sauravmaheshkar/initialization/reports/A-Gentle-Introduction-To-Weight-Initialization-for-Neural-Networks--Vmlldzo2ODExMTg)
[3](https://towardsdatascience.com/kaiming-he-initialization-in-neural-networks-math-proof-73b9a0d845c4/)
[4](https://www.machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/)
[5](https://arxiv.org/pdf/1606.00305.pdf)
[6](https://arxiv.org/pdf/2104.03693.pdf)
[7](http://proceedings.mlr.press/v70/safran17a/safran17a.pdf)
[8](https://dl.acm.org/doi/pdf/10.5555/3305890.3305989)
[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC5859950/)
[10](https://ieeexplore.ieee.org/document/10995028/)
[11](https://arxiv.org/html/2507.11436v1)
[12](https://ieeexplore.ieee.org/document/7934087/)
[13](https://www.semanticscholar.org/paper/bbd8ff9ac6b66820e30b34d75c551f0e6d1fde42)
[14](https://arxiv.org/abs/2403.01123)
[15](https://ieeexplore.ieee.org/document/10041931/)
[16](https://arxiv.org/pdf/2209.14778.pdf)
[17](https://openreview.net/forum?id=162TqkUNPO&noteId=NBgzhQ3fCm)
[18](https://arxiv.org/pdf/2404.03176.pdf)
[19](https://www.semanticscholar.org/paper/d7fc9c4a2dae141d825d620147629cec86b5da9b)
[20](https://www.semanticscholar.org/paper/16cb0d4803ba7fbcd383d245ddad348993da84d9)
[21](https://arxiv.org/pdf/2307.06555v4.pdf)
[22](https://arxiv.org/pdf/1502.01852.pdf)
[23](https://arxiv.org/pdf/2406.14936.pdf)
[24](http://arxiv.org/pdf/2110.12246.pdf)
[25](https://arxiv.org/pdf/2306.00651.pdf)
[26](http://arxiv.org/pdf/2405.03777.pdf)
[27](https://www.machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)
[28](https://www.superannotate.com/blog/activation-functions-in-neural-networks)
[29](https://www.sciencedirect.com/science/article/abs/pii/S0925231217315758)
[30](https://cas.cnu.ac.kr/wp-data/international_conference/%5B2024%5D%20Implementation%20of%20Activation%20Functions%20using%20various%20approximation%20methods.pdf)
[31](https://koreascience.kr/article/JAKO201733348350618.page)
[32](https://www.semanticscholar.org/paper/24da6180db314619060d7b8fc798390f0c7a139a)
[33](http://ieeexplore.ieee.org/document/8099908/)
[34](https://www.semanticscholar.org/paper/4ade95ddc8e6948e4094d0454b2429930123d4ca)
[35](http://ieeexplore.ieee.org/document/7872790/)
[36](https://www.semanticscholar.org/paper/89e930c841f70e4949f4e531dbca8948fe51ddef)
[37](https://www.semanticscholar.org/paper/7f8b247b492d8a3e7cbdd8158a8b4a296497f45c)
[38](http://arxiv.org/pdf/2103.11642.pdf)
[39](https://arxiv.org/pdf/1502.03167.pdf)
[40](https://arxiv.org/pdf/1603.09025.pdf)
[41](https://arxiv.org/pdf/2209.08898.pdf)
[42](https://arxiv.org/html/2306.16999)
[43](https://arxiv.org/pdf/1806.02375.pdf)
[44](https://arxiv.org/pdf/2106.03970.pdf)
[45](http://papers.neurips.cc/paper/6770-train-longer-generalize-better-closing-the-generalization-gap-in-large-batch-training-of-neural-networks.pdf)
[46](http://d2l.ai/chapter_convolutional-modern/batch-norm.html)
[47](https://openreview.net/pdf?id=BJe55gBtvH)
[48](https://arxiv.org/pdf/2403.08408.pdf)
[49](https://www.ijcai.org/proceedings/2019/0595.pdf)
[50](https://www.semanticscholar.org/paper/3da5618c8fcc7a0f0203f4d2e5d2f0364f5ea45f)
[51](https://www.mdpi.com/2075-4418/13/20/3234)
[52](https://ieeexplore.ieee.org/document/9669990/)
[53](https://www.semanticscholar.org/paper/aa87035617832d192e3e3dcb27b297ca2c70c5c2)
[54](https://www.semanticscholar.org/paper/edccc38cbd8765c658b3880facec76e9f4a8ee5c)
[55](http://link.springer.com/10.1007/978-3-030-16841-4_35)
[56](http://link.springer.com/10.1007/978-3-030-01424-7_44)
[57](https://dl.acm.org/doi/10.1145/3123266.3123353)
[58](https://ieeexplore.ieee.org/document/10010588/)
[59](https://ieeexplore.ieee.org/document/11076511/)
[60](https://arxiv.org/pdf/1805.07477.pdf)
[61](https://arxiv.org/pdf/2102.01351.pdf)
[62](http://arxiv.org/pdf/1711.09485.pdf)
[63](https://arxiv.org/pdf/2401.09018.pdf)
[64](http://arxiv.org/pdf/1910.00780.pdf)
[65](https://arxiv.org/ftp/arxiv/papers/1707/1707.05425.pdf)
[66](https://arxiv.org/pdf/2012.07356.pdf)
[67](http://arxiv.org/pdf/2311.15947.pdf)
[68](https://paravisionlab.co.in/resnet/)
[69](https://www.scnsoft.com/blog/imagenet-challenge-2017-expectations)
[70](https://www.vizuaranewsletter.com/p/resnet-the-architecture-that-changed)
[71](https://en.wikipedia.org/wiki/ImageNet)
[72](https://arxiv.org/pdf/2410.16711.pdf)
[73](https://arxiv.org/html/2405.01725v1)
[74](https://image-net.org/challenges/LSVRC/2017/)
[75](https://www.sciencedirect.com/science/article/pii/S2666521225000857)
[76](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
