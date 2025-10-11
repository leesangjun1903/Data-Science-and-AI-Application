# TopologyNet: Topology-Based Deep Convolutional Neural Networks for Biomolecular Property Predictions

**핵심 주장 및 주요 기여**  
TopologyNet은 3D 생체분자 구조의 복잡한 기하학적·생물학적 정보를 해소하기 위해 **Element Specific Persistent Homology (ESPH)**를 도입하고, 이를 **다채널 이미지 표현**으로 변환하여 딥러닝에 적용한 최초의 프레임워크이다. ESPH는 분자의 위상 정보를 1차원 바코드로 압축하면서 원소별 화학 정보를 보존한다. 이러한 다채널 1D 이미지 표현을 1D 합성곱 신경망(CNN)에 통합하여 **구조-기능 관계**(단백질-리간드 결합 친화도, 돌연변이 안정성 변화)를 예측하며,  
- **단일 작업 TopologyNet**: ESPH + CNN → 최첨단 예측 성능 달성.  
- **다중 작업 MT-TCNN**: 유사 예측 과제를 동시에 학습하여 소규모·노이즈 많음 데이터의 **일반화 성능**을 크게 향상.  

**1. 해결하고자 하는 문제**  
전통적인 물리 기반·경험 기반 모델은  
- 기하학적 세부 정보 과부하: 3D 좌표·거리·각도 등 방대한 수치가 딥러닝에 부적합  
- 학습 데이터 부족·노이즈 큰 생체 데이터 세트에서 딥러닝 과적합  

**2. 제안 방법**  
2.1. ESPH 기반 다채널 1D 이미지 표현  
- 각 분자(단백질·리간드·돌연변이 잔기) 원자 유형별(Betti-0,1,2) 바코드를 계산하여,  
  - Birth(Vₑᵦ), Death(Vₑ_d), Persistence(Vₑₚ) 벡터로 변환  
  - 필터 반경 구간을 픽셀로, 원소·차원·이벤트별 채널 구성 → 1×n×m 다채널 1D 이미지  
- 수식 예:  

```math
    V_d^{i} = \#\{\,\text{death}_j\in [\,\tfrac{i-1}{n}L,\tfrac{i}{n}L)\}
``` 

2.2. 1D CNN 아키텍처  
- **합성곱층**: 윈도우 크기 w의 필터 F∈ℝ^{w×m}으로 로컬 패턴 학습  
- **최적화**: 확률적 경사하강법(SGD) + 모멘텀  

$$\theta_{t+1}=\theta_t-\eta\nabla L+\alpha(\theta_t-\theta_{t-1})$$  

- **정규화**: 드롭아웃과 배깅(서로 다른 모델 평균화)으로 과적합 방지  
- **다중 작업 MT-TCNN**: 공유 합성곱층 후 과제별 분기(branch) → 고차원 일반화 표현 학습  

**3. 모델 구조**  
- 입력: 1×n×m 다채널 토폴로지 이미지  
- 층 구성:  
  - Conv1D → Pooling → … → Flatten → Fully-Connected → 출력(회귀)  
- MT-TCNN: 초반 합성곱층 공유 → 두 개의 Fully-Connected 분기 → Globular/Membrane mutation 예측  

**4. 성능 향상**  
- **단백질-리간드 결합 친화도**(PDBBind 2007 Core):  
  - Pearson $$R_p=0.826$$, RMSE=1.37 → 기존 최고 기법 대비 향상[1]
- **단일 작업 돌연변이 안정성**(S350):  
  - TopologyNet-MP-1: $$R_p=0.74$$, RMSE=1.07  
  - TopologyNet-MP-2(+진화·서열 정보): $$R_p=0.81$$, RMSE=0.94$$ → MTL 통한 일반화 개선[1]
- **다중 작업 막단백질 돌연변이**(M223):  
  - MT-TCNN: $$R_p=0.52$$ → 비MT 대비 8.3% 향상[1]

**5. 한계**  
- **고차원 위상 정보**(Betti-1,2) 계산 비용  
- 1D 컨볼루션 필터 **이동 불변성 가정 부재**: 필터 공유가 이상적이지 않음  
- **데이터 편중**: 희소 과제에서는 여전히 성능 한계  

**6. 일반화 성능 향상 전략**  
- **다중 작업 학습**: 소규모 과제 간 공동 표현 학습으로 과적합 저감  
- **드롭아웃+배깅**: 모델 간 예측 평균화로 경계 케이스 일반화  
- **공유 필터**: 거리 스케일 전반의 특징 패턴 학습 → 경계 영역 처리 능력 강화  

**7. 향후 연구 영향 및 고려사항**  
- **위상 표현 확장**: 다차원 위상 불변량(Persistent Cohomology, Zigzag Persistence) 통합 연구  
- **이동 불변성 완화**: 로컬 연결층(Local-Connected Layer) 대안 검토  
- **융합 학습**: 토폴로지·그래프 신경망·분자 역학 시뮬레이션 결합  
- **실험 설계**: 큰 스케일 실험·크로스 도메인 벤치마크 구축으로 일반화 검증  
- **효율 최적화**: GPU 가속화 Persistent Homology 알고리즘 개발  

TopologyNet은 **위상수학**과 **딥러닝**을 융합한 혁신적 접근으로, 생체분자 예측 분야의 새로운 패러다임을 제시하며, 향후 **토폴로지 기반 AI** 연구 방향을 선도할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/30f10250-2a75-4c22-9ea5-89982eb963fe/TopologyNet-Topology-based-deep-convolutional-neural-networks-for-biomolecular-property-predictions.pdf)
