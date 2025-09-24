# Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation | Few-Shot Learning, Learning-to-Learn Approach (Meta-Learning)

## 1. 핵심 주장 및 주요 기여  
본 논문은 **도메인 간 격차(domain shift 문제로 인해 기존 메트릭 기반 few-shot 분류기가 **미지의 도메인**에서는 일반화 성능이 크게 저하된다는 점을 지적하고, 이를 해결하기 위해 **feature-wise transformation layer**를 제안한다.  
- 이 레이어는 중간 층의 특징 맵(feature map)에 **채널별 가우시안 분포 기반의 어파인 변환**을 적용하여 학습 단계에서 다양한 도메인 분포를 시뮬레이션한다[1].  
- 또한, 하이퍼파라미터(어파인 변환의 표준편차)를 **메타 학습(learning-to-learn)** 방식으로 최적화하여, 여러 출발 도메인(seen domains)으로부터 학습한 변환 파라미터가 미지 도메인에서도 잘 작동하도록 설계하였다[1].  
- 이 기법을 MatchingNet, RelationNet, GNN 등 **세 가지 메트릭 기반 모델**에 적용한 결과, mini-ImageNet→CUB, Cars, Places, Plantae 등 **5개 도메인 전이** 실험에서 일관된 성능 향상을 보였다[1].  

## 2. 문제 정의, 제안 방식, 모델 구조, 성능 개선 및 한계  
### 2.1 해결하고자 하는 문제  
Meta-learning 기반의 메트릭 방식 few-shot 분류기는  
1) 지원 집합(support set)의 소수 샘플과  
2) 질의 집합(query set)의 샘플을,  
공통된 백본 인코더 E와 거리 함수 M으로 분류하지만,  
학습 도메인과 테스트 도메인의 **특징 분포(feature distribution)** 차이가 크면 성능이 급락한다[1].  

### 2.2 제안하는 방법  
#### Feature-Wise Transformation Layer  
중간 특징 맵 $$z\in\mathbb{R}^{C\times H\times W}$$에 대해 채널별로  

$$
\gamma_c\sim\mathcal{N}(1,\ \mathrm{softplus}(\theta_{\gamma,c})),\quad
\beta_c\sim\mathcal{N}(0,\ \mathrm{softplus}(\theta_{\beta,c}))
$$  

를 샘플링한 뒤,  

$$
\hat z_{c,h,w} = \gamma_c\cdot z_{c,h,w} + \beta_c
$$  

를 수행하여, 다양한 스케일과 편향이 섞인 특징 분포를 학습 단계에 생성한다[1].  

#### Learning-to-Learn으로 하이퍼파라미터 최적화  
학습 시 매 반복마다  
1. 일부 도메인을 **pseudo-seen**으로, 다른 일부를 **pseudo-unseen**으로 삼고  
2. pseudo-seen 과제에서 모델 파라미터 $$(\theta_e,\theta_m)$$을 업데이트  
3. pseudo-unseen 과제에서 분류 손실 $$L_{\text{pu}}$$을 계산  
4. 이 손실에 대해 $$\theta_f=\{\theta_\gamma,\theta_\beta\}$$를 경사하강법으로 갱신  
하는 절차를 통해, 실제 미지 도메인 일반화 성능을 직접 최적화한다[1].  

### 2.3 모델 구조  
- 백본: ResNet-10  
- 메트릭 기반 분류기: MatchingNet, RelationNet, GNN  
- 각 잔차 블록의 배치 정규화 뒤에 feature-wise transformation layer 삽입  

### 2.4 성능 향상  
- mini-ImageNet→미지 도메인 전이 시, 모든 모델이 1–5% absolute 개선  
- 5-way 5-shot 기준, GNN + FT: 70.84%→73.94% (Places) 등 일관된 향상[1][Table 1]  
- 다중 도메인 학습(leave-one-out) 및 learning-to-learn 적용 시 추가 2–4% 향상[1][Table 2]  

### 2.5 한계  
- 변환 층 하이퍼파라미터($$\theta_\gamma,\theta_\beta$$) 최적화 공간이 매우 고차원(약 1920차원)  
- learning-to-learn 방식도 전역 최적해를 보장하진 않으며, 계산 비용이 큼  
- 도메인 간 차이가 극단적일 경우, 단순 어파인 변환만으로는 분포 격차를 완전히 메우기 어려움  

## 3. 일반화 성능 향상 관점  
- **다양한 특징 분포 시뮬레이션**: feature-wise 변환이 학습 단계에 배치돼, 메트릭 함수 M이 특정 도메인 분포에 과적합되지 않음[1].  
- **메타 최적화**: 하이퍼파라미터를 가상의 미지 도메인 손실로 직접 갱신하여, 실제 일반화 성능을 초깃값 단계부터 반영[1].  
- t-SNE 시각화에서, 원 도메인과 미지 도메인 샘플들이 변환 층 적용 시 더 가깝게 클러스터됨을 확인[1][Fig. 3].  

## 4. 향후 연구에 미치는 영향 및 고려 사항  
- **범용 도메인 일반화**: feature-wise 변환과 메타 학습 결합은 다른 비전 작업(예: 객체 검출, 분할)에서도 미지 상황 대응 연구에 영감을 줄 수 있음.  
- **경량화·효율화**: 고차원 하이퍼파라미터 최적화 부담을 줄이기 위한 차원 축소 기법이나 적응형 스케줄링 연구 필요.  
- **조합 연구**: 어파인 변환 외에 주파수 도메인 조작, 어텐션 조정, 적대적 증강 등과의 하이브리드가 일반화 성능을 더 끌어올릴 가능성.  

결론적으로, 본 논문은 **feature-wise affine 변환**과 **learning-to-learn**을 결합해 few-shot 분류기의 **도메인 간 일반화** 문제를 체계적으로 해결했으며[1], 후속 연구에서 다양한 도메인 일반화 기법과 접목돼 응용될 수 있는 토대를 제공한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c47c228d-14f9-4d8b-bf80-1745f1efe73f/2001.08735v3.pdf
[2] https://www.semanticscholar.org/paper/17b99c60d6b2fdd656af6a7661b8d6af05255792
[3] https://arxiv.org/abs/2406.16422
[4] https://link.springer.com/10.1007/s00521-022-07897-9
[5] https://arxiv.org/abs/2203.02270
[6] https://arxiv.org/abs/2208.11021
[7] https://ieeexplore.ieee.org/document/10025014/
[8] https://link.springer.com/10.1007/s10489-023-04948-z
[9] https://ieeexplore.ieee.org/document/9709997/
[10] https://h-j-han.github.io/papers/cross_domain_few_shot_classification_via_learned_feature_wise_transformation/
[11] https://proceedings.neurips.cc/paper_files/paper/2023/file/bbb7506579431a85861a05fff048d3e1-Paper-Conference.pdf
[12] https://arxiv.org/html/2504.06608v1
[13] https://par.nsf.gov/biblio/10170545-cross-domain-few-shot-classification-via-learned-feature-wise-transformation
[14] https://openaccess.thecvf.com/content/ICCV2021/papers/Das_On_the_Importance_of_Distractors_for_Few-Shot_Classification_ICCV_2021_paper.pdf
[15] https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720120.pdf
[16] https://sanghani.cs.vt.edu/research/publications/2020/cross-domain-few-shot-classification-via-learned-feature-wise-transformation.html
[17] https://faculty.ucmerced.edu/mhyang/papers/iclr2020_few_shot.pdf
[18] https://openaccess.thecvf.com/content/ACCV2020/papers/Guan_Large-Scale_Cross-Domain_Few-Shot_Learning_ACCV_2020_paper.pdf
[19] https://arxiv.org/html/2406.16422v1
[20] https://arxiv.org/pdf/2203.02270.pdf
[21] https://www.ijcai.org/proceedings/2024/0607.pdf
[22] https://yonsei.elsevierpure.com/en/publications/cross-domain-few-shot-classification-via-learned-feature-wise-tra-2
[23] https://icml.cc/virtual/2025/poster/46472
[24] https://arxiv.org/abs/2001.08735
[25] https://www.sciencedirect.com/science/article/abs/pii/S0950705122002805
[26] https://ieeexplore.ieee.org/document/9999670/
[27] https://ieeexplore.ieee.org/document/9423301/
[28] http://arxiv.org/pdf/2401.15834.pdf
[29] http://arxiv.org/pdf/2208.11021.pdf
[30] https://arxiv.org/pdf/2007.08790.pdf
[31] http://arxiv.org/pdf/2011.00179.pdf
[32] https://arxiv.org/pdf/2109.12548.pdf
[33] https://arxiv.org/abs/2311.02392v1
[34] https://arxiv.org/pdf/2401.13987.pdf
[35] https://arxiv.org/pdf/2003.09338.pdf
[36] https://arxiv.org/pdf/1904.04232.pdf
[37] https://github.com/hytseng0509/CrossDomainFewShot
[38] https://www.sciencedirect.com/science/article/abs/pii/S1077314223001170
[39] https://openreview.net/forum?id=SJl5Np4tPr
[40] https://scispace.com/pdf/cross-domain-few-shot-classification-via-learned-feature-2keabyoe7j.pdf
