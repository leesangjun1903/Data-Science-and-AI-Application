# Efficiently Modeling Long Sequences with Structured State Spaces

## 1. 핵심 주장과 주요 기여  
이 논문은 **Structured State Space sequence model (S4)** 을 통해 기존 RNN, CNN, Transformer 계열 모델들이 10,000 스텝 이상의 장기 의존성(Long-Range Dependencies, LRD)을 다루기 어렵다는 문제를 해결하고자 한다.  
- **핵심 주장**: 연속시간 상태공간모델(SSM)을 새롭게 매개변수화하고 효율적인 수치 알고리즘을 결합하면, 계산·메모리 복잡도를 선형 수준 $$O(N+L)$$으로 줄이면서도 매우 긴 시퀀스에서도 우수한 성능을 얻을 수 있다.  
- **주요 기여**  
  1. **NPLR(Normal Plus Low‐Rank) 파라미터화**: HiPPO 행렬을 정상 행렬(normal)과 저랭크 보정(low-rank) 항으로 분해해, 안정적인 대각화와 Woodbury 항등식 적용을 가능케 함.  
  2. **SSM 생성함수(Generating Function) 활용**: 시퀀스 필터 $$K = (C A^i B)_{i=0}^{L-1}$$ 대신, 복소수 주파수 영역의 생성함수 $$\hat K(z) = C\,(I - A z)^{-1}B$$를 FFT를 통해 빠르게 평가.  
  3. **Cauchy 커널 알고리즘**: Woodbury 보정 후 생성함수 계산을 Cauchy 행렬 곱셈으로 귀결시켜, 수치적으로 안정적이고 준선형 시간 복잡도로 구현.  

## 2. 해결 과제, 제안 방법, 모델 구조, 성능 및 한계  

### 2.1 해결 과제  
- 기존 SSM 기반 LSSL(Linear State Space Layer)은 시퀀스 길이 $$L$$과 상태 차원 $$N$$에 대해 $$O(N^2L)$$ 연산량, $$O(NL)$$ 메모리를 요구해 실제 활용 불가.  
- RNN/CNN/Transformer 각 방식의 개선판조차도 $$L\gg10^4$$인 경우 장기 의존성 처리와 효율성에서 한계.  

### 2.2 제안 방법  
1. **HiPPO 행렬 초기화**  

$$
     A_{n,k} =
     \begin{cases}
       -\sqrt{(2n+1)(2k+1)} & n > k,\\
       -(n+1)                 & n=k,\\
       0                      & n < k
     \end{cases}
   $$

2. **NPLR 분해**  

$$
     A = V\Lambda V^* - P\,Q^*,
   $$
   
   여기서 $$V$$ 유니터리, $$\Lambda$$ 대각, $$P,Q$$ 저랭크 벡터.  
3. **이산화(bilinear discretization)**  

$$
     A_d = (I - \tfrac{\Delta}{2}A)^{-1}(I + \tfrac{\Delta}{2}A),\quad
     B_d = (I - \tfrac{\Delta}{2}A)^{-1}\,\Delta B.
   $$

4. **생성함수 및 FFT**  

$$
     \hat K(z) = \sum_{i=0}^{L-1} (C A_d^i B_d)\,z^i = C\,(I - A_d z)^{-1}(I - A_d^L z^L)\,B_d.
   $$

5. **Woodbury 보정**  

$$
     (I - A_d z)^{-1}
     = \bigl(I - (\Lambda - P Q^*)\,z\bigr)^{-1}
     = D(z) - D(z)P\,(I + Q^*D(z)P)^{-1}Q^*D(z),
   $$
   
   $$D(z)$$ 대각 행렬.  

6. **Cauchy 커널**  

$$\hat K(z)$$ 계산을 $$\bigl(\omega_i - \lambda_j\bigr)^{-1}$$ 형태의 Cauchy 행렬 곱으로 환원해 준선형 시간 수치 알고리즘 활용.  

### 2.3 모델 구조  
- 한층(S4 레이어)마다 상태 차원 $$N$$인 SSM을 $$H$$ 피처 병렬로 복제하고, 그 뒤에 위치별(feed-forward) 선형 혼합 및 비선형 활성화 삽입.  
- 일반 Transformer와 동일한 입출력 텐서 형태 $$(B, L, H)$$를 유지하며, 비선형 층을 쌓아 깊이 있는 구조 구성.  

### 2.4 성능 향상  
- **Long Range Arena**: 평균 정확도 86.1%로 기존 SoTA 대비 25% 이상, 가장 어려운 Path-X(길이 16K) 과제 최초 해결.  
- **Raw Speech Classification**: 16K 길이의 원신호만으로 98.3% 정확도, MFCC 기반 CNN·RNN보다 우수.  
- **CIFAR-10 및 WikiText-103 생성**: 비자연스러운 2D·토큰 의존성 없이도 2.85 bits/dim, perplexity 20.95 달성. 생성 속도는 Transformer 대비 60× 빠름.  
- **수치 효율성**: LSSL 대비 최대 30× 빠르고 400× 적은 메모리, Transformer·Performer와 경쟁력 있는 훈련·추론 효율.  

### 2.5 한계  
- **하드웨어 최적화 필요**: 현재 GPU 구현은 PyKeOps 기반으로 최적화 여지 있음.  
- **언어모델 성능 격차**: 대형 Transformer 대비 언어 모델링(perplexity)에서 여전히 소폭 열세.  
- **고차원 데이터 확장**: 2D·비정형 시퀀스(영상, 그래프)로 일반화 연구 필요.  

## 3. 일반화 성능 및 향후 고려점  
- **적응적 샘플링**: 연속시간 해석 덕분에 입력 샘플링률 변경 시 재학습 없이 대응 가능(예: 0.5× 주파수).  
- **도메인 지식 최소화**: 이미지·음성·텍스트·시계열 등 다양한 모달리티에 동일 아키텍처 적용 가능, 도메인 특화 전처리 불필요.  
- **표현력 및 규제**: HiPPO 기반 초기화는 학습 안정성과 일반화에 결정적 역할. 추가적인 저차원 보강이나 스파스 규제와 결합해 더 강건한 모델링 가능성.  

## 4. 향후 연구 방향 및 고려 사항  
- **고차원 SSM 확장**: 2D 윈도잉, 스파스 패턴 결합으로 영상·비디오 장기 의존성 모델링 연구.  
- **하드웨어 최적화**: 전용 CUDA Cauchy 커널 및 FMM 알고리즘 도입으로 실시간 대규모 배포.  
- **하이브리드 모델**: S4와 Transformer·CNN·GNN 결합해 LRD 처리와 로컬 패턴 탐지 균형.  
- **이론적 분석 심화**: HiPPO 이론을 넘어 일반 연속시간 시스템의 안정성·표현력 해석.  

---  
S4는 장기 의존성 문제에서 이론적·실용적 돌파구를 제공하며, 범용 시퀀스 모델 연구의 새로운 방향을 제시한다. 현 수준의 한계를 극복하기 위해 효율적 구현, 고차원 확장, 하이브리드 아키텍처 설계가 후속 과제로 남아 있다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5d6632a1-834a-45b8-bd33-0fa09a195859/2111.00396v3.pdf)
