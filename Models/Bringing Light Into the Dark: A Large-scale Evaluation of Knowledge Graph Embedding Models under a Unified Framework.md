# Bringing Light Into the Dark: A Large-scale Evaluation of Knowledge Graph Embedding Models under a Unified Framework

### 1. 핵심 주장과 주요 기여

**"Bringing Light Into the Dark: A Large-scale Evaluation of Knowledge Graph Embedding Models under a Unified Framework"**는 **지식 그래프 임베딩 모델(KGEM)의 재현성 위기와 공정한 비교의 어려움**을 중앙 문제로 삼는 연구이다.[1][2][3][4]

**핵심 기여:**[2][5][6][1]

**재현성 연구(Reproducibility Study):** 21개의 지식 그래프 임베딩 모델을 PyKEEN 프레임워크에서 재구현하고, 이전에 발표된 논문의 최적 하이퍼파라미터로 원래 결과를 재현할 수 있는지 체계적으로 검증했다. 연구진은 어떤 결과는 원래 하이퍼파라미터로 재현 가능했으며, 어떤 것은 다른 하이퍼파라미터로만 재현 가능했고, 어떤 것은 전혀 재현 불가능했는지를 명확히 제시했다.

**대규모 벤치마킹 연구(Benchmarking Study):** 4개의 표준 데이터셋(FB15k, FB15k-237, WN18, WN18RR)에 대해 **24,804 GPU 시간**을 투입하여 수천 개의 실험을 수행했다.[7][5][1][2]

**통합 프레임워크:** 모델 구조(interaction model), 훈련 접근 방식(training approach), 손실 함수(loss function), 역관계 명시적 모델링(explicit modeling of inverse relations)이라는 **4개의 핵심 구성 요소**를 포함한 통합된 분석 프레임워크를 개발했다.[7][1]

***

### 2. 해결하고자 하는 문제

논문이 직면한 **두 가지 핵심 문제:**[8][1][7]

**첫 번째: 재현성 위기**
- 이전에 발표된 논문들에서 동일한 모델과 데이터셋 조합에 대해 서로 다른 결과가 보고되는 사례가 있었다.
- 소스 코드의 미공개 또는 다양한 프로그래밍 언어와 프레임워크의 사용으로 인한 불일치가 발생했다.
- 하이퍼파라미터의 불명확한 명시로 인해 재현이 불가능했다.

**두 번째: 신규성 검증의 어려움**
- 새로운 모델 아키텍처가 주장하는 성능 개선이 실제로는 **훈련 방식, 하이퍼파라미터 값, 전처리 방법(예: 역관계의 명시적 모델링)**으로 인한 것인지 구별하기 어려웠다.
- 기준 모델(baseline)이 적절히 최적화되면 최신 모델과 경쟁할 수 있음이 이전 연구에서 보여졌으나, 이것이 체계적으로 검증되지 않았다.[9][10]

***

### 3. 제안하는 방법 및 모델 구조

#### A. 통합 프레임워크의 4가지 핵심 구성 요소

**1) 상호작용 모델(Interaction Model)**

지식 그래프 임베딩 모델은 삼중항(triple) (h, r, t)의 타당성을 평가하는 점수 함수 $$f(h, r, t)$$를 정의한다. 평가된 주요 모델들:[1][7]

| 모델 | 점수 함수 | 특징 |
|------|---------|------|
| **TransE** | $$\lVert h + r - t \rVert$$ | 번역 기반, 대칭 관계 모델링 불가 |
| **TransH** | $$\lVert g_{r,1}(h) + r - g_{r,2}(t) \rVert$$ | 관계별 하이퍼플레인 사영 |
| **DistMult** | $$\langle h \odot r \odot t \rangle$$ | 대칭 관계만 모델링 |
| **ComplEx** | $$\text{Re}(\langle h, r, t \rangle)$$ | 복소수 공간, 비대칭 관계 모델링 |
| **RotatE** | 복소 공간에서 회전 변환 | 합성 관계 패턴 모델링 |
| **ConvE** | 합성곱 신경망 기반 | 2층 신경망 및 합성곱 연산 |
| **CP** | $$\sum_k h_k \cdot r_k \cdot t_k$$ | 텐서 분해 방식 |

**2) 훈련 접근 방식(Training Approach)**

논문은 **폐쇄 세계 가정(Closed World Assumption, LCWA)**과 **확률적 폐쇄 세계 가정(Stochastic LCWA, sLCWA)**의 두 가지를 비교했다.[7][1]

**LCWA(Local Closed World Assumption):**
- 각 (h, r) 쌍에 대해 모든 가능한 꼬리 엔티티를 동시에 평가
- 1-N 점수 계산에 효율적
- 공식: $$T^-(h,r) = \{(h,r,t') : t' \in E \setminus \{t : (h,r,t) \in K\}\}$$[7]

**sLCWA(Stochastic LCWA):**
- 배치 단위로 음성 샘플링(negative sampling) 수행
- 메모리 효율적
- 균일 샘플링과 베르누이 음성 샘플링 지원[1][7]

**3) 손실 함수(Loss Function)**

세 가지 주요 손실 함수를 평가했다:

**Pointwise Loss Functions:**
- **Pairwise Margin Ranking Loss (MRL):** $$L = \sum_{(h,r,t)\in K} \max(0, f(h,r,t^-) - f(h,r,t) + \gamma)$$
- 양성 삼중항과 음성 삼중항 간의 마진 최대화

**Ranking Loss:**
- 예측된 순위에 기반한 손실

**Cross Entropy Loss (CEL):**
- 다중 클래스 분류 문제로 프레이밍
- LCWA 훈련 루프에서 모든 가능한 꼬리 엔티티에 대한 확률 계산[1][7]

**4) 역관계의 명시적 모델링(Inverse Relations)**

각 삼중항 (h, r, t)에 대해 역삼중항 $$(t, r^{-1}, h)$$를 추가로 훈련:[7][1]

- 관계 임베딩 공간을 암시적으로 2배로 증가
- 모델 성능과 계산 효율성에 긍정적 영향
- 구현: 각 학습 반복에서 음성 삼중항과 함께 역삼중항도 생성

#### B. 평가 메트릭(Evaluation Metrics)

링크 예측 작업의 성능을 평가하기 위해 다음의 순위 기반 메트릭을 사용:[11][12][8]

1. **Mean Rank (MR):** 평균 순위 (낮을수록 좋음)
2. **Mean Reciprocal Rank (MRR):** 평균 역순위 = $$\frac{1}{|P_{test}|}\sum_{\lambda \in P_{test}} \frac{1}{\text{rank}(\lambda)}$$
3. **Hits@N:** 상위 N개 내 정확도 (N = 1, 3, 10)
4. **Adjusted Mean Rank (AMR):** 데이터셋 크기 차이를 고려한 조정 평균 순위[13]

***

### 4. 성능 향상 및 주요 발견

#### A. 재현성 연구 결과[8][1][7]

- 21개 모델 중 상당수가 원래 하이퍼파라미터로는 재현 불가능
- 대체 하이퍼파라미터로 많은 결과 재현 가능
- 이는 **정확한 하이퍼파라미터 명시의 중요성** 강조

#### B. 벤치마킹 연구의 핵심 발견[8][1][7]

**1) 모델 아키텍처만으로는 불충분**

> 모델의 성능은 **아키텍처뿐만 아니라 훈련 접근 방식, 손실 함수, 역관계 명시적 모델링의 조합**에 의해 결정된다.[8][7]

- 여러 아키텍처가 신중하게 구성되면 최신 기술과 경쟁 가능한 성과 달성

**2) 역관계 모델링의 영향**

역관계 명시적 모델링 포함 여부에 따른 성능 변화:
- FB15k-237 데이터셋: **큰 성능 향상** 관찰
- WN18RR 데이터셋: **중간 정도의 향상**
- 모델과 훈련 설정에 따라 다양한 효과

**3) 손실 함수의 영향**

- **Cross Entropy Loss (CEL):** LCWA 훈련에서 높은 성능
- **Margin Ranking Loss (MRL):** sLCWA 훈련에 효과적
- 손실 함수와 훈련 접근 방식의 **상호작용 중요**

**4) 최적 구성의 발견**

특정 데이터셋과 모델에 따른 최적 구성:
- FB15k-237: RotatE + LCWA + CEL + 역관계 포함
- WN18RR: ComplEx 또는 RotatE + sLCWA + MRL

#### C. 데이터셋별 분석[1][7]

| 데이터셋 | 엔티티 수 | 관계 수 | 삼중항 수 | 특성 |
|----------|---------|--------|----------|------|
| **FB15k** | 14,951 | 1,345 | 592,213 | 주요 관계 패턴: 합성, 비대칭 |
| **FB15k-237** | 14,541 | 237 | 310,116 | 역관계 제거, 더 도전적 |
| **WN18** | 40,943 | 18 | 141,442 | 작은 관계 세트, 대칭 관계 많음 |
| **WN18RR** | 40,943 | 11 | 93,003 | 역관계 제거, 더 현실적 |

---

### 5. 모델의 일반화 성능과 한계

#### A. 일반화 성능에 관한 발견

**1) 전이 학습 가능성**

논문에서 직접 다루지는 않으나, 결과는 **신중한 하이퍼파라미터 조정을 통해 모델 간 성능 차이가 축소**될 수 있음을 시사한다.[8][7][1]

**2) 데이터셋 간 성능 일관성**

- 특정 모델/구성이 FB15k와 FB15k-237에서 모두 우수한 성능 달성
- 그러나 완전히 다른 특성의 데이터셋(WN18)에서는 다른 최적 구성 필요

**3) 외삽 vs 전이(Generalization Challenge)**

논문의 결과는 다음을 시사한다:
- **도메인 내 일반화:** 유사한 KG 특성을 가진 데이터셋 간 전이 가능
- **도메인 간 일반화:** 다른 구조의 KG로의 전이는 제한적

#### B. 한계 및 미해결 문제

**1) 귀납적 링크 예측의 부재**[5][6][1]

논문은 **귀납적 설정(inductive setting)**을 다루지 않는다. 즉:
- 훈련 중 보지 못한 새로운 엔티티에 대한 예측 불가
- 실제 KG는 지속적으로 새로운 엔티티 추가
- 임베딩 기반 방법의 확장성 제한

**2) 시간적 KG의 배제**[5][1]

- 동적 KG(temporal KG)에 대한 평가 미수행
- 시간적 정보를 포함한 모델 구조 미분석

**3) 다중 모달 정보**

- 텍스트, 이미지 등 다중 모달 정보와의 통합 미검토
- 최신 멀티모달 KG 임베딩 방법 미포함

**4) 계산 복잡도 분석의 부족**

- 모델별 계산 시간과 메모리 사용량에 대한 상세 분석 제한적
- 규모 확장성(scalability) 평가 불충분

***

### 6. 최근 연구에 미치는 영향 및 향후 고려 사항(2023-2025)

#### A. 직접적 영향[14][15][16][17][18][19][20][21]

**1) 귀납적 링크 예측 연구의 활성화**

논문의 한계를 극복하기 위해 최근 연구들이 **귀납적 설정에서의 KG 임베딩**에 집중:
- **GraiL/NeuralLP 기반 방법:** 부분 그래프 구조를 이용한 엔티티 독립적 학습[16][17][18]
- **TGraiL (2023):** 위상(topological) 구조와 의미 정보를 결합하여 미보지 못한 엔티티의 일반화 능력 향상[17][16]
- **SiaILP (2023):** 시암 신경망을 이용한 경로 기반 귀납적 링크 예측[21]

**2) 전이 학습 연구의 확대**

- **SCR (2024):** 공통 지식 그래프에서 훈련하여 다양한 그래프 작업으로 전이 가능한 기초 모델 개발[19]
- 표준 KG에서 학습한 표현이 다른 도메인의 그래프에 전이 가능함을 입증

**3) 베이스라인 재평가 트렌드**

- 본 논문 이후 새로운 모델 제안 시 **이전 기준 모델의 신중한 하이퍼파라미터 최적화**가 표준 관행으로 정착
- 성능 비교의 공정성이 주요 평가 기준

#### B. 향후 연구에서 고려할 점[22][20][19][21]

**1) 벤치마크 데이터셋의 현실성 재검토**[20]

최근 연구(2025)는 표준 벤치마크가 **비현실적 평가**를 할 수 있음을 지적:
- 이상적인 폐쇄 세계 가정(CWA)이 실제 불완전한 KG와 맞지 않음
- 새로운 현실적 평가 프레임워크 필요

**2) 다양한 다운스트림 작업으로의 확대**[22][19]

- 링크 예측 최적화만으로는 다운스트림 작업 성능 보장 불가
- 특정 응용(질문 답변, 추천 시스템)에 맞춘 평가 프레임워크 필요

**3) 불확실성 모델링**[14]

최근 연구들은 KG 임베딩에서 **확률적 의미론**을 도입:
- 각 삼중항의 신뢰도를 확률로 모델링
- 전역 일관성 보장

**4) 세만틱과 구조의 통합**[18][16][17]

- 순수 구조 정보만으로는 불충분
- 엔티티 타입, 관계 의미론, 온톨로지 정보의 명시적 활용 필요

**5) 인덕티브 추론에서의 규모 확장성**[18]

- 엔티티-독립적 모델링으로 미보지 못한 엔티티에 대응
- 삼중각형 뷰 그래프 변환과 그래프 신경망의 결합

**6) 사전 학습 패러다임의 탐색**[22]

- 링크 예측만을 목표로 한 사전 학습이 다운스트림 작업에 최적인지 재검토
- 다양한 사전 학습 목표 탐색 필요

***

### 결론

"Bringing Light Into the Dark" 논문은 **지식 그래프 임베딩 분야의 메타-과학적 위기**를 체계적으로 해결했다. 단순 모델 아키텍처만으로 성능이 결정되지 않으며, **훈련 방식, 손실 함수, 하이퍼파라미터, 역관계 모델링이 동등하게 중요**함을 입증했다.[7][1][8]

최근(2023-2025) 후속 연구들은 이 논문의 한계를 직시하고 **귀납적 설정, 다중 모달 정보 통합, 현실적 벤치마크 재정의, 다운스트림 작업 성능 향상**에 집중하고 있다. 향후 연구자들은 이 논문의 방법론적 엄격성을 유지하면서도 더욱 다양한 시나리오와 응용에 초점을 맞춰야을 유지하면서도 더욱 다양한 시나리오와 응용에 초점을 맞춰야 할 것으로 판단된다.

[1](https://www.scribd.com/document/718412962/Bringing-Light-Into-the-Dark-A-Large-scale-Evaluation-of-Knowledge-Graph-Embedding-Models-under-a-Unified-Framework)
[2](https://pykeen.github.io)
[3](https://www.dbs.ifi.lmu.de/~tresp/papers/2006.13365.pdf)
[4](https://github.com/pykeen/benchmarking)
[5](https://jmlr.csail.mit.edu/papers/volume22/20-825/20-825.pdf)
[6](https://jmlr.org/papers/v22/20-825.html)
[7](https://backend.orbit.dtu.dk/ws/files/262633429/Bringing_Light_Into_the_Dark_A_Large_scale_Evaluation_of_Knowledge_Graph_Embedding_Models_under_a_Unified_Framework.pdf)
[8](https://arxiv.org/abs/2006.13365)
[9](https://arxiv.org/pdf/2310.14899.pdf)
[10](https://arxiv.org/html/2402.13630)
[11](http://arxiv.org/pdf/2402.06098.pdf)
[12](https://mcml.ai/publications/hbg+22/)
[13](https://aclanthology.org/2021.emnlp-main.769.pdf)
[14](https://www.aclweb.org/anthology/2021.naacl-main.68.pdf)
[15](http://arxiv.org/pdf/2312.09219.pdf)
[16](https://www.aclweb.org/anthology/2021.naacl-main.221.pdf)
[17](https://www.nature.com/articles/s41598-023-48616-1)
[18](http://uclab.khu.ac.kr/resources/publication/J_304.pdf)
[19](https://arxiv.org/html/2410.12609v2)
[20](https://arxiv.org/pdf/2504.08970.pdf)
[21](https://arxiv.org/abs/2312.10293)
[22](https://aclanthology.org/2024.repl4nlp-1.11.pdf)
[23](https://www.aclweb.org/anthology/2020.emnlp-demos.22.pdf)
[24](http://arxiv.org/pdf/2307.01933.pdf)
[25](https://arxiv.org/pdf/2406.01759.pdf)
[26](https://aclanthology.org/2023.findings-emnlp.580.pdf)
[27](https://arxiv.org/pdf/2306.08302.pdf)
[28](https://github.com/nju-websoft/OpenEA)
[29](https://www.sciencedirect.com/science/article/abs/pii/S0950705124001035)
[30](https://arxiv.org/abs/1908.06543)
[31](https://proceedings.kr.org/2023/45/kr2023-0045-liu-et-al.pdf)
[32](https://migalkin.github.io/publication/2020-06-01-pykeen)
[33](http://www.ecologyandsociety.org/vol26/iss1/art15/ES-2020-12156.pdf)
[34](https://arxiv.org/ftp/arxiv/papers/2104/2104.03550.pdf)
[35](https://wellcomeopenresearch.org/articles/6-69/v3/pdf)
[36](https://tidsskrift.dk/nja/article/download/140104/184096)
[37](https://www.mdpi.com/1660-4601/18/2/624/pdf)
[38](https://www.mdpi.com/2076-0752/10/3/56/pdf)
[39](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/B63BBC62692479F3B2118BDA07B800BD/S0814062624000120a.pdf/div-class-title-experiments-with-a-dark-pedagogy-learning-from-through-temporality-climate-change-and-species-extinction-and-ghosts-div.pdf)
[40](https://www.mdpi.com/2076-3417/14/7/2945/pdf?version=1711866467)
[41](https://dl.acm.org/doi/10.1145/3589334.3645430)
[42](https://www.vldb.org/pvldb/vol15/p633-kochsiek.pdf)
[43](https://journals.sagepub.com/doi/10.3233/NAI-240731)
[44](https://dl.acm.org/doi/10.14778/3494124.3494144)
[45](https://openreview.net/pdf/3c6fb8702426fd9bee11292be39dc1e1392fa6d4.pdf)
[46](https://scholar.google.com.ec/citations?user=h25eyTIAAAAJ&hl=ko)
[47](https://ceur-ws.org/Vol-3741/paper04.pdf)
[48](http://arxiv.org/pdf/2409.14857.pdf)
[49](http://arxiv.org/pdf/2303.12816.pdf)
[50](https://arxiv.org/pdf/1903.03772.pdf)
[51](https://arxiv.org/pdf/2311.00115.pdf)
[52](https://arxiv.org/pdf/2206.02963.pdf)
[53](https://arxiv.org/pdf/1910.06708.pdf)
[54](https://arxiv.org/pdf/2110.03789.pdf)
[55](https://openreview.net/pdf?id=HkgEQnRqYQ)
[56](https://arxiv.org/pdf/2105.10488.pdf)
[57](https://arxiv.org/html/2507.00965v1)
[58](https://sebd2024.unica.it/papers/paper04.pdf)
[59](https://www.semantic-web-journal.net/system/files/swj3320.pdf)
[60](https://arxiv.org/html/2404.03499v1)
[61](http://arxiv.org/pdf/2205.01331.pdf)
[62](https://arxiv.org/pdf/1803.04042.pdf)
[63](https://arxiv.org/pdf/2109.08935.pdf)
[64](https://arxiv.org/pdf/1809.09414.pdf)
[65](http://arxiv.org/pdf/2205.03876.pdf)
[66](https://arxiv.org/pdf/2108.09628.pdf)
[67](https://arxiv.org/pdf/2006.07060.pdf)
[68](http://arxiv.org/pdf/2309.05681.pdf)
[69](https://www.biorxiv.org/content/10.1101/2023.01.10.523485v1.full)
[70](https://arxiv.org/pdf/2309.03773.pdf)
[71](https://bonndoc.ulb.uni-bonn.de/xmlui/bitstream/handle/20.500.11811/11122/7280.pdf;jsessionid=AB1DF880D99390EC4FEF4735CFC65199?sequence=2)
[72](https://pykeen.github.io/kgem-meta-review/)
[73](https://arxiv.org/pdf/2102.07200.pdf)
[74](http://arxiv.org/pdf/2308.00081.pdf)
[75](https://arxiv.org/pdf/1903.11406.pdf)
[76](http://arxiv.org/pdf/2407.15906.pdf)
[77](https://en.wikipedia.org/wiki/Knowledge_graph_embedding)
[78](https://jens-lehmann.org/research-areas/representation-learning-in-knowledge-graphs/)
[79](https://arxiv.org/pdf/2508.05587.pdf)
[80](https://dbs.uni-leipzig.de/file/EAGERpreprint.pdf)
[81](https://graph-learning-benchmarks.github.io/assets/papers/glb2022/An_Open_Challenge_for_Inductive_Link_Prediction_on_Knowledge_Graphs.pdf)
[82](https://ieeexplore.ieee.org/document/9601281)
[83](https://journals.sagepub.com/doi/10.3233/SW-212959)
[84](https://www.sciencedirect.com/science/article/abs/pii/S0893608024010323)
[85](https://scholar.google.com/citations?user=h25eyTIAAAAJ&hl=en)
