# GTM: The Generative Topographic Mapping

**핵심 주장 및 주요 기여**  
Generative Topographic Mapping(GTM)은 Kohonen의 Self-Organizing Map(SOM)을 확률론적(latent variable) 모델로 재정의하여,  
-  명확한 우도함수(likelihood)를 갖는 **확률 밀도 모델**을 제시  
-  EM 알고리즘을 통해 **최적화** 가능한 비선형 잠재 변수 모델을 도입  
-  SOM의 학습률·이웃 함수 스케줄링 문제, 수렴성·공간 배치 보장의 부재 한계를 극복  

***

## 1. 해결하고자 하는 문제  
전통적 SOM은  
- 비용 함수 부재로 최적화 방향 불명확  
- 수렴 보장 및 이웃 함수 스케줄링 이론적 근거 부족  
- 확률 밀도 정의 불가  
이로 인해 지도(mapping) 품질과 해석 가능성에 한계가 존재  

***

## 2. 제안 방법  
GTM은 잠재 변수 $$x\in\mathbb{R}^L$$에서 데이터 공간 $$t\in\mathbb{R}^D$$로의 매핑을 확률 모델로 정의한다.  

1) **잠재 공간 샘플링**  

$$
   p(x)=\frac{1}{K}\sum_{i=1}^K \delta(x - x_i)
   $$  
   
   - 규칙적 격자점 $$x_i$$에서의 델타 함수 혼합  

2) **매핑 함수**  

$$
   y(x;W)=W\,\phi(x),\quad \phi(x)=(\phi_1(x),\dots,\phi_M(x))
   $$  
   
   - 방사형 기저 함수(RBF) $$\phi_j(x)$$ 활용  
   - $$W\in\mathbb{R}^{D\times M}$$  

3) **관측 모델**  

$$
   p(t|x,W,\beta)=\biggl(\frac{\beta}{2\pi}\biggr)^{\frac{D}{2}}
   \exp\Bigl(-\frac{\beta}{2}\|t - y(x;W)\|^2\Bigr)
   $$  

4) **데이터 우도 및 EM 최적화**  

$$
   p(t|W,\beta)=\frac{1}{K}\sum_{i=1}^K p(t|x_i,W,\beta),\quad
   \mathcal{L}=\sum_{n=1}^N\ln p(t_n|W,\beta)
   $$  
   
   - E-step: 책임도(responsibility)

$$
     r_{in}=\frac{p(t_n|x_i)}{\sum_j p(t_n|x_j)}
     $$  
   
   - M-step:  

$$\displaystyle W^{\text{new}}$$는  
     

$$
     \Phi^\top G\,\Phi\,W^\top = \Phi^\top R\,T
     $$  
     
  선형 방정식으로 재추정  

$$
     \beta^{\text{new}}=\frac{1}{ND}\sum_{n,i} r_{in}\|t_n - y(x_i)\|^2
     $$  

***

## 3. 모델 구조 및 학습 흐름  
1. **초기화**: PCA를 이용해 $$W$$와 $$\beta$$ 설정  
2. **반복 학습**  
   - E-step: 각 $$t_n$$에 대한 posterior $$r_{in}$$ 계산  
   - M-step: $$W$$, $$\beta$$ 갱신  
   - 수렴 시까지 우도 증가 확인  
3. **시각화**: 잠재 공간에서 posterior 평균 $$\langle x|t_n\rangle=\sum_i r_{in}x_i$$ 및 모드 표시  

***

## 4. 성능 향상 및 한계  
- **성능 향상**  
  - SOM 대비 **우도 기반 수렴** 보장  
  - **매개변수 선택**(기저 함수 폭, 격자 수 등)이 명확  
  - **소프트 할당**(soft assignment)으로 더 세밀한 클러스터링  
  - **연산 비용**: 데이터–센터 거리 계산 복잡도는 SOM과 유사  

- **한계**  
  - 대규모 데이터·고차원 잠재 공간에서 $$K$$ 증가 시 계산량 급증  
  - 기저 함수 수·격자 해상도 선택에 따른 메모리 사용량 이슈  
  - 복잡한 분포 모델링 시 RBF 기반 매핑 한계  

***

## 5. 일반화 성능 향상 가능성  
- **정규화·사전분포**: $$W$$에 대한 가우시안 사전분포 $$\lambda$$ 추가로 과적합 억제  
- **혼합 GTM**: 여러 GTM 모델 혼합으로 다봉분포(multi-modal) 대응  
- **베이지안 확장**: 우도 대신 사후분포 전 범위 고려하는 완전 베이지안 추론  
- **기저 함수 학습**: RBF 폭·위치도 MAP 또는 EM으로 최적화  

이로써 새로운 데이터 분포에 대한 **범용성·강건성**을 제고할 수 있음  

***

## 6. 향후 연구 영향 및 고려사항  
- **확장성 연구**: 대규모·고차원 데이터에 대한 계산 효율화(스파스 기저, 인크리멘털 EM)  
- **심층 매핑**: 딥러닝 기반 비선형 함수 $$y(x)$$로 일반화  
- **불완전·결측치 데이터 처리**: GTM의 확률모델 특성 활용한 결측치 보완  
- **도메인 특화 응용**: 의료영상·고차원 과학 데이터 시각화 및 클러스터링에 응용  
- **하이브리드 모델**: GTM과 트랜스포머, 그래프 신경망 등 최신 아키텍처 결합  

이러한 방향들이 GTM을 **차세대 잠재 변수 모델**로 발전시키는 핵심 고려사항이 될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/82e6f3ab-b414-49ef-83a3-b72e2e2ae3d5/bishop-gtm-ncomp-98.pdf)
