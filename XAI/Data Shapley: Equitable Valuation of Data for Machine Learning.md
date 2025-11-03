# Data Shapley: Equitable Valuation of Data for Machine Learning

### 1. 핵심 주장과 주요 기여 요약

**Data Shapley** 논문의 중심 주장은 **게임이론의 Shapley value를 머신러닝의 데이터 가치 평가에 적용**하여, 각 훈련 데이터가 모델 성능에 기여하는 정도를 공정하게 정량화할 수 있다는 것입니다.[1]

주요 기여는 다음과 같습니다:[1]

- **공정한 데이터 가치 평가 프레임워크** 제시: 기존의 Leave-One-Out(LOO) 방법의 한계를 극복하는 원칙적 접근법 도입
- **세 가지 공정성 공리(Equitable Properties) 만족**: 널 플레이어 공리, 대칭성 공리, 선형성 공리를 동시에 충족하는 유일한 방법
- **실용적 계산 방법 개발**: Monte Carlo 방법과 그래디언트 기반 방법을 통해 실제 대규모 데이터셋과 신경망에 적용 가능
- **다양한 응용 가능성 입증**: 이상치 탐지, 오염된 라벨 감지, 데이터 수집 전략 수립 등

***

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능

#### 2.1 해결하는 문제

데이터의 가치를 정량화하는 것이 핵심 문제입니다. 논문은 다음 두 가지 질문을 제시합니다:[1]

1. 학습 알고리즘 A와 성능 지표 V에 대해, 각 훈련 데이터 $$(x_i, y_i)$$의 공정한 가치 척도는 무엇인가?
2. 이러한 가치를 실제 환경에서 효율적으로 계산하는 방법은 무엇인가?

**기존 LOO 방법의 한계**: 완벽하게 같은 데이터 두 개를 훈련 집합에 포함할 경우, k-NN 분류기는 하나를 제거해도 성능이 변하지 않아 둘 다 0의 값을 받습니다. 이는 실제로 두 데이터가 모두 중요할 수 있다는 직관과 맞지 않습니다.[1]

#### 2.2 공정한 데이터 가치 평가를 위한 공리

데이터 가치 함수 $$\varphi$$는 다음 세 가지 성질을 만족해야 합니다:[1]

**공리 1 (Null Player Property)**: 데이터 i가 어떤 부분집합에 추가되어도 성능에 변화를 주지 않으면 $$\varphi_i = 0$$
$$\text{for all } S \subseteq D - \{i\}, \quad V(S) = V(S \cup \{i\}) \Rightarrow \varphi_i = 0$$

**공리 2 (Symmetry Property)**: 두 데이터 i, j가 어떤 부분집합에 추가될 때 동일한 성능 변화를 만들면 $$\varphi_i = \varphi_j$$
$$\text{for all } S \subseteq D - \{i,j\}, \quad V(S \cup \{i\}) = V(S \cup \{j\}) \Rightarrow \varphi_i = \varphi_j$$

**공리 3 (Linearity/Additivity Property)**: 성능 지표가 합산 가능하면(예: $$V = V_1 + V_2$$), 데이터의 가치도 합산 가능해야 함
$$\varphi_i(V + W) = \varphi_i(V) + \varphi_i(W)$$

#### 2.3 Data Shapley 수식

**Proposition 2.1**에서 위 세 공리를 모두 만족하는 유일한 형태의 함수를 제시합니다:[1]

$$\varphi_i = C \sum_{S \subseteq D - \{i\}} \frac{V(S \cup \{i\}) - V(S)}{\binom{n-1}{|S|}}$$

여기서:
- $$S$$: 데이터 i를 포함하지 않는 D의 부분집합
- $$V(S \cup \{i\}) - V(S)$$: 데이터 i의 **한계 기여도(Marginal Contribution)**
- $$\binom{n-1}{|S|}$$: 크기 $$|S|$$인 부분집합의 개수(역가중)
- $$C$$: 정규화 상수

이 공식은 게임이론의 **Shapley value**와 동일하며, "모든 가능한 부분집합 시나리오에서 데이터의 평균적 한계 기여도"를 의미합니다.[1]

**해석**: 각 데이터 포인트의 가치는 그것이 훈련 데이터의 모든 가능한 순열(permutation)에서 얼마나 성능을 향상시키는지에 대한 가중 평균입니다.[1]

#### 2.4 계산 복잡성 및 근사 방법

정확한 Data Shapley 계산은 $$2^n$$의 지수 시간 복잡도를 가지므로 실제 적용이 불가능합니다. 따라서 논문은 두 가지 근사 방법을 제시합니다:[1]

**방법 1: Truncated Monte Carlo Shapley (TMC-Shapley)**

$$\varphi_i = \mathbb{E}_{\pi \sim \Pi}[V(S_i^\pi \cup \{i\}) - V(S_i^\pi)]$$

여기서 $$\pi$$는 모든 n! 순열 중 균등하게 선택된 순열, $$S_i^\pi$$는 순열에서 데이터 i 이전의 데이터들의 집합입니다.[1]

**알고리즘 개요** (Algorithm 1):
1. 훈련 데이터의 임의의 순열을 샘플링
2. 순열을 따라 순차적으로 스캔하며 각 데이터의 한계 기여도 계산
3. 성능이 전체 데이터셋의 성능과 "성능 허용도(Performance Tolerance)" 내에 도달하면 이후 한계 기여도를 0으로 설정(Truncation)
4. 수렴할 때까지 반복하여 평균값 계산

**장점**: 
- 몬테카를로 샘플링으로 불편 추정(Unbiased Estimate) 제공
- Truncation으로 계산 시간 대폭 단축(부트스트랩 분산을 기반으로 설정 가능)
- 일반적으로 3n개의 몬테카를로 샘플로 수렴[1]

**방법 2: Gradient Shapley (G-Shapley)**

신경망과 같이 계산 비용이 높은 모델을 위해 제시된 방법입니다(Algorithm 2):[1]

1. 임의의 순열에서 각 데이터를 하나씩 순차적으로 처리
2. 각 데이터에 대해 **단일 에포크(한 번의 그래디언트 스텝)만 실행**
3. 그래디언트 업데이트 전후의 성능 변화를 한계 기여도로 계산

$$\theta_j^t \leftarrow \theta_{j-1}^t - \alpha \nabla_\theta L(\pi^t[j]; \theta_{j-1}^t)$$

여기서 $$\alpha$$는 학습률입니다.

**근사 이론**: 완전하게 학습된 모델과의 오류 분석은 Appendix D에 제시되며, 경험적으로 우수한 근사를 제공합니다.[1]

---

### 3. 일반화 성능 향상 관련 내용 (중점)

#### 3.1 고가치 데이터 선택을 통한 성능 개선

**질병 예측 실험** (Figure 1):[1]

유방암 및 피부암 예측 과제에서:
- 높은 Data Shapley 값을 가진 데이터부터 제거 시 성능이 급격히 저하
- LOO는 무작위 제거와 거의 같은 수준의 성능 저하(즉, 가치 있는 데이터 식별 실패)
- 낮은 Shapley 값을 가진 데이터 제거 시 오히려 성능이 향상 → 이상치나 오염 데이터 제거 효과

**새로운 데이터 수집 전략** (Figure 1c, 1d):[1]
- 높은 Shapley 값을 가진 기존 환자와 **유사한 새로운 환자**를 추가하면 모델 성능이 무작위 추가보다 훨씬 빠르게 개선
- 반대로 낮은 Shapley 값 환자와 유사한 환자 추가 시 성능 개선 효과 미미 또는 악화

**적용 메커니즘**:
1. 훈련 데이터의 Shapley 값 계산
2. Random Forest 회귀 모델을 훈련시켜 "특성 → Shapley 값" 매핑 학습
3. 후보 데이터 풀의 각 데이터에 대해 예상 Shapley 값 추정
4. 높은 예상 값을 가진 데이터 선택하여 추가 수집

#### 3.2 합성 데이터 실험 (Figure 2)

**선형 및 비선형 관계 분석**:[1]

50차원 가우시안 데이터 생성:
- **선형 관계**: 로지스틱 회귀 모델 → Shapley 방법이 LOO보다 우수
- **3차 다항식**: 로지스틱 회귀 vs. 신경망(숨겨진 층 1개)
- **핵심 관찰**: 비선형 모델에서 고가치 데이터 식별 성능 격차 확대

**일반화 성능 의미**:
- 특정 모델 유형에서 중요한 데이터는 다른 모델에서는 해로울 수 있음
- 따라서 Shapley 값은 **모델-의존적**이며, 목표 모델에 특화된 데이터 선택 가능
- 이는 모델 일반화 능력을 개선하되, 해당 모델의 특성에 맞춘 데이터 선택이 중요함을 시사

#### 3.3 라벨 노이즈 및 데이터 품질 (Figure 3, 5)

**라벨 오염 실험** (Figure 3):[1]

세 가지 데이터셋/모델 조합:
1. **Spam 분류** (Naive Bayes): 20% 라벨 반전 → TMC-Shapley로 가장 효과적으로 오염 데이터 탐지
2. **꽃 분류** (Multinomial Logistic Regression, Inception-V3 특성): 10% 오염
3. **Fashion MNIST** (CNN): 10% 오염

**결과**: 
- 최저 Shapley 값 데이터부터 검사하면 오염 데이터를 가장 빠르게 식별
- LOO는 로지스틱 회귀에서는 수행하지만 신경망에서는 무작위와 유사
- Shapley 값이 음수인 데이터(모델에 해로운 데이터)가 대부분 오염됨

**데이터 품질-가치 관계** (Figure 5a):[1]
- 백색 잡음 추가 비율 증가에 따라 데이터의 평균 Shapley 값 감소
- 깨끗한 데이터와 잡음이 있는 데이터의 Shapley 값 차이 확대
- **해석**: 모델이 자동으로 데이터 품질을 반영한 가치 평가 수행

#### 3.4 그룹 Shapley와 공정성

**병원 재입원 예측** (Figure 5b):[1]

환자를 인구통계학적 특성(성별, 인종, 나이)으로 146개 그룹으로 분할:
- 고령층 그룹이 높은 Shapley 값 → 재입원 예측에 더 정보적
- 소수 인종 그룹의 낮은 값 → 데이터 표현 부족 문제 가시화
- **공정성 함의**: 데이터 가치 평가가 모델 편향을 간접적으로 드러낼 수 있음

---

### 4. 한계점

#### 4.1 계산 비용
- 정확한 Shapley 값: $$O(2^n)$$ 복잡도로 대규모 데이터 불가능
- TMC-Shapley: 수렴까지 $$O(n^2 \log n)$$ 정도의 비용
- 대규모 신경망: 120시간/4개 GPU(Inception 기반 CNN) 필요[1]

#### 4.2 G-Shapley의 근사 오류
- 단일 에포크로 계산하므로 완전히 수렴한 모델과의 차이 존재
- Hyperparameter 조정 필요(멀티 에포크 학습보다 높은 학습률)

#### 4.3 문제-의존성
- 데이터 값은 알고리즘, 성능 지표, 다른 데이터에 모두 의존
- "보편적 데이터 가치"는 정의 불가능
- 컨텍스트 변경 시 재계산 필요

#### 4.4 개인정보 및 사회적 함의
- 논문은 "사람들이 정확히 Shapley 값만큼 보상받아야 한다"고 주장하지 않음
- 프라이버시, 개인적 연관성 등 정량화되지 않는 가치는 미반영[1]

***

### 5. 최신 연구 기반 영향과 향후 고려 사항

#### 5.1 Data Shapley의 후속 연구 및 개선

**1) 확장성 개선**[2][3][4]

- **In-Run Data Shapley** (2024, ICLR Outstanding Runner-Up): 단일 모델 훈련 과정 중에 데이터 값을 계산하여 계산 비용을 거의 0으로 감소 → 기초 모델 사전학습 단계에 처음으로 적용 가능[4]
- **CHG Shapley** (2024): Hardness와 Gradient를 결합한 유틸리티 함수로 계산 복잡도를 $$O(n^2 \log n)$$에서 단일 모델 학습 수준으로 개선[2]
- **LossVal** (2024): 손실 함수에 자가 가중 메커니즘 삽입하여 훈련 중 데이터 중요도 점수 계산, 라벨 노이즈와 특성 노이즈를 동시에 처리[3]

**2) 이론적 개선**[5][6]

- **Beta Shapley** (2022): Efficiency 공리를 완화하여 더 안정적이고 노이즈에 강한 데이터 가치 평가 제시[6]
- **Asymmetric Data Shapley** (2024): 데이터 내 고유한 구조를 반영한 구조-인식 데이터 가치 평가[5]

**3) 응용 확장**[7][8][9]

- **Class-wise Shapley (CS-Shapley)** (2022): 클래스 내/클래스 간 기여도를 구분하여 오염된 데이터 탐지 강화[9]
- **데이터 파이프라인 통합** (DataScope, 2022): 전처리, 특성 추출을 포함한 end-to-end ML 파이프라인에 Shapley 값 적용[7]
- **의료 이미지 분석** (2021): 흉부 X-ray 분류에 k-NN 기반 Shapley 값 적용, 오염 데이터 탐지 및 데이터 공유 인센티브 설계[8]

#### 5.2 현재 연구 동향

**Explainable AI(XAI)에서의 중심 역할**[10][11]

- SHAP(SHapley Additive exPlanations) 기반 모델 해석이 업계 표준화
- shapiq 라이브러리(2024): Shapley value와 상위 차수 특성 상호작용(Shapley Interactions) 통합 계산 플랫폼
- Vision Transformer, 대형 언어 모델까지 확장된 적용

**의료 및 금융 도메인 활용**[12][13][14]

- 당뇨병 관리, 금융 부실 예측 등에서 Shapley 기반 특성 중요도 분석
- COVID-19 HLA 변이 분석, 암 유전체 분석 등 생물정보학 응용[13]

#### 5.3 향후 연구 시 고려할 점

**1) 데이터 특성과 모델 선택의 상호작용**

- Data Shapley 값이 매우 모델-의존적이므로, 목표 모델 선택이 중요
- 다양한 모델 아키텍처(CNN, Transformer, Tree-based 등)에서의 일반화 성능 비교 필요
- 모델 앙상블의 맥락에서 집계된 Shapley 값 계산 방안 연구

**2) 계산 효율성**[3][4][2]

- **대규모 데이터셋**: In-Run 방법이나 CHG Shapley 등 O(1) 또는 선형 시간 알고리즘 우선 고려
- **GPU/분산 환경**: TMC-Shapley의 병렬화 수준 최적화
- **데이터 스트리밍**: 점진적으로 업데이트 가능한 Shapley 값 계산

**3) 데이터 품질과 노이즈 처리**

- 라벨 노이즈와 특성 노이즈 동시 처리 방법론 정교화
- 약한 라벨(Weak Label) 또는 불균형 데이터셋에서의 Shapley 값 안정성
- 아웃라이어 vs. 유용한 엣지 케이스 구분

**4) 공정성 및 윤리**[15]

- 그룹 Shapley를 통한 데이터 편향 가시화 및 완화 전략
- 데이터 마켓플레이스에서 공정한 보상 체계 설계
- 개인정보 보호와 데이터 가치 평가의 균형

**5) 전이 가능성(Transferability)**

- 한 모델/작업에서 계산한 Shapley 값이 다른 모델/작업으로 전이 가능한지 체계적 연구 필요
- Class-wise Shapley의 모델 간 전이성이 부분적으로 입증되었으나, 더 광범위한 조사 필요

**6) 실제 데이터 수집 및 큐레이션**

- 고가치 데이터 특성 자동 파악 및 신규 데이터 수집 가이드 라인 개발
- 액티브 러닝(Active Learning)과의 통합으로 적응적 데이터 수집 최적화
- 기초 모델(Foundation Models) 사전학습 데이터 필터링 및 선택[4]

***

### 요약

Data Shapley는 게임이론의 Shapley value를 머신러닝의 데이터 가치 평가에 처음 도입한 획기적 논문으로, 공정성 공리를 만족하는 유일한 방법론을 제시합니다. **Monte Carlo와 Gradient 기반 근사 알고리즘**으로 실제 신경망 규모 데이터에 적용 가능하며, 이상치 탐지, 라벨 오염 식별, 효율적 데이터 수집 전략 수립에 있어 기존 LOO 방법보다 우수한 성능을 보입니다.

최근 연구는 **계산 효율성 개선**(In-Run Shapley, CHG Shapley), **이론적 확장**(Beta/Asymmetric Shapley), **실제 적용 확대**(의료, 금융, 대형 모델)에 집중하고 있습니다. 향후 연구는 대규모 데이터와 모델에 대한 실시간 계산, 데이터 편향과 공정성 측면의 심화, 기초 모델 시대에 맞춘 데이터 큐레이션 자동화에 초점을 맞출 것으로 예상됩니다.

---

#### 참고 자료

 Ghorbani, A., & Zou, J. (2019). Data Shapley: Equitable Valuation of Data for Machine Learning. *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97.[1]

 Mazumder, S. et al. (2022). Data Debugging with Shapley Importance over End-to-End Machine Learning Pipelines. *arXiv:2204.11131*.[7]

 Jia, R. et al. (2021). Trustworthy machine learning for health care: scalable data valuation with the Shapley value. *ACM Conference on Fairness, Accountability, and Transparency*.[8]

 Review on Trending ML Techniques for Type 2 Diabetes. *MDPI* (2024). [공개된 웹 자료][12]

 HLA Variants and COVID-19 Severity Identification. *IEEE Transactions* (2024).[13]

 Towards Data Valuation via Asymmetric Data Shapley. *arXiv:2411.00388* (2024).[5]

 Beta Shapley: a Unified and Noise-reduced Data Valuation Framework. *arXiv:2110.14049* (2022).[6]

 A Comprehensive Study of Shapley Value in Data Analytics. *VLDB* 18(9): 3077-3092 (2025).[15]

 CHG Shapley: Efficient Data Valuation and Selection. *arXiv:2406.11730* (2024).[2]

 LossVal: Efficient Data Valuation for Neural Networks. *ICML* (2024).[3]

 Data Shapley in One Training Run. *arXiv:2406.11011*, ICLR 2025 Outstanding Runner-Up (2024).[4]

 Class-wise Shapley Values for Data Valuation in Classification. *NeurIPS* 35 (2022).[9]

 shapiq: Shapley Interactions for Machine Learning. *NeurIPS* 37 (2024).[10]

 The Shapley Value in Machine Learning (Survey). *IJCAI* (urvey). *IJCAI* (2022).[29]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/42debe87-1fab-4093-aabd-58c9518a8ba8/ghorbani19c.pdf)
[2](https://arxiv.org/pdf/2406.11730.pdf)
[3](https://arxiv.org/html/2412.04158v1)
[4](https://arxiv.org/abs/2406.11011)
[5](https://arxiv.org/html/2411.00388)
[6](https://arxiv.org/abs/2110.14049)
[7](https://arxiv.org/abs/2204.11131)
[8](https://dl.acm.org/doi/10.1145/3450439.3451861)
[9](https://proceedings.neurips.cc/paper_files/paper/2022/hash/df334022279996b07e0870a629c18857-Abstract-Conference.html)
[10](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb3a9313405e2d4175a5a3cfcd49999b-Abstract-Datasets_and_Benchmarks_Track.html)
[11](https://www.ijcai.org/proceedings/2022/778)
[12](https://www.mdpi.com/2227-9709/11/4/70)
[13](https://ieeexplore.ieee.org/document/10903468/)
[14](https://www.mdpi.com/2306-5729/7/11/160)
[15](https://www.vldb.org/pvldb/vol18/p3077-xie.pdf)
[16](https://link.springer.com/10.1007/s41024-024-00445-z)
[17](https://www.tandfonline.com/doi/full/10.1080/01431161.2023.2217982)
[18](https://dx.plos.org/10.1371/journal.pone.0286829)
[19](https://xlink.rsc.org/?DOI=D2CP04428E)
[20](https://pubs.acs.org/doi/10.1021/acs.jcim.4c02414)
[21](https://pmc.ncbi.nlm.nih.gov/articles/PMC11515487/)
[22](https://arxiv.org/abs/2107.07436)
[23](https://dx.plos.org/10.1371/journal.pone.0277975)
[24](http://arxiv.org/pdf/2401.12683.pdf)
[25](https://arxiv.org/pdf/2110.02484v1.pdf)
[26](https://pmc.ncbi.nlm.nih.gov/articles/PMC9683574/)
[27](https://ijsrm.net/index.php/ijsrm/article/download/5762/3566/16944)
[28](https://jy1559.tistory.com/7)
[29](https://arxiv.org/html/2502.09969v3)
