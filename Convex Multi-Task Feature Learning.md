# Convex Multi-Task Feature Learning

**핵심 요약**  
Convex Multi-Task Feature Learning은 여러 관련 작업(task)을 동시에 학습하면서 *공통의 희소한(feature) 표현*을 자동으로 학습하는 알고리즘을 제안한다. 이 방법은 비–볼록(non-convex) 문제를 볼록(convex) 최적화 문제로 변환하고, 교대로(alternating) 최적화하는 알고리즘을 통해 전역 최적해를 보장한다.

## 1. 서론 및 문제 설정  
- **다중 작업 학습(Multi-Task Learning)**  
  - 여러 관련된 예측 작업(예: 서로 다른 사용자별 선호도 모델링)을 동시에 학습  
  - 서로 다른 작업들이 *공통의 표현(잠재 특징)*을 공유한다고 가정  
- **목표**  
  1. 각 작업별 예측 함수 $$f_t(x)$$ 학습  
  2. 작업 간에 공유되는 *희소(sparse)*한 특징 집합 $$h_i(x)$$ 자동 학습  
  3. 학습된 특징의 수는 정규화 파라미터로 제어  
- **모델화**  

$$
    f_t(x) = \sum_{i=1}^N a_{it}\,h_i(x), 
    \quad
    h_i(x)=\langle u_i,\,x\rangle,\;
    U=[u_1,\ldots,u_d]\in\mathbb{R}^{d\times d},\;A=(a_{it})\in\mathbb{R}^{d\times T}.
  $$
  - $$U$$ 열은 직교(orthogonal)  
  - $$A$$ 행(row)의 $$\ell_2$$-노름을 $$\ell_1$$-노름으로 합친 $$(2,1)$$-노름을 사용해 “몇 개의 행만 0이 아니도록” 유도  



## 2. 비–볼록 문제에서 볼록 문제로의 변환  
- **원래 문제(비–볼록)**  

$$
    \min_{U\in O_d,\;A\in\mathbb{R}^{d\times T}}
      \sum_{t=1}^T \sum_{i=1}^m L(y_{ti},\,\langle a_t,\,U^\top x_{ti}\rangle)
      \;+\;\gamma\,\|A\|_{2,1}.
  $$
- **등가 볼록 문제 도출**  
  - 추가 변수 $$D\in S^d_+$$ 도입, $$\mathrm{trace}(D)\le1$$, $$\mathrm{range}(W)\subseteq\mathrm{range}(D)$$ 제약  
  - $$W=UA$$ 대입 후      

$$
      \min_{W,\;D\succeq0}
        \sum_{t,i} L(y_{ti},\,\langle w_t,\,x_{ti}\rangle)
        \;+\;\gamma\sum_{t}\langle w_t,\,D^+w_t\rangle.
    $$
  - **정리**: 원래 비–볼록 문제는 이 볼록 문제와 정확히 동일 최적값을 갖는다.  

## 3. 최적화 알고리즘  
### 3.1. ε-perturbed 교대 최적화  
1. **$$W$$-고정, $$D$$ 최적화**  

$$
     D\;=\;\arg\min_{D\succeq0,\mathrm{tr}(D)\le1}
       \sum_t\langle w_t,\,D^{-1}w_t\rangle
       +\epsilon\,\mathrm{tr}(D^{-1})
     \quad\Longrightarrow\;
     D=\frac{(WW^\top+\epsilon I)^{1/2}}{\mathrm{tr}(WW^\top+\epsilon I)^{1/2}}
   $$

2. **$$D$$-고정, $$W$$ 최적화**  
   - 각 작업 $$t$$에 대해  
     $$\sum_i L(y_{ti},\,\langle w,\,x_{ti}\rangle)+\gamma\,\langle w,\,D^{-1}w\rangle$$  
     → 일반 2-노름 정규화 문제로 분리 가능  
3. **수렴**  
   - $$\epsilon>0$$인 경우 전역 최적해로 수렴 증명  
   - $$\epsilon\to0$$ 시 원본 볼록 문제 최적해 획득  

### 3.2. 커널 확장 (비선형 특징 학습)  
- **표현 정리**: 최적 $$w_t$$는 훈련 샘플의 커널 기저 표현  
- 입력 맵 $$\phi(x)$$ 대신 훈련점 기저로 차원 축소 → 유크라우셜(square-root) 교대 최적화 알고리즘 유사하게 적용  
- 커널 행렬 계산만으로 비선형 공통 특징 학습 가능  

## 4. 실험 결과 요약  
1. **합성 데이터**  
   - 선형·비선형 기저에서 진짜 공통 특징 회복 성공  
   - 작업 수 증가 시 테스트 오차 감소, 특징 추정 오차 감소  
   - 정규화 파라미터 $$\gamma$$ 증가 시 학습 특징 수 감소  
2. **리얼 데이터**  
   - **소비자 선호(conjoint) 실험**: 공통 특징→ ‘기술 사양 vs. 가격’ 벡터 학습  
   - **학교 성적 예측(school data)**: 학생 능력(밴드)↔가장 큰 공통 특징  
   - **피부 질환 분류(dermatology)**: 작업 간 상관성 낮아, 동시 학습 성능 변화 없음  

## 5. 의의 및 향후 연구  
- **새로운 정규화**: $$(2,1)$$-노름으로 다중 작업 간 ‘공통 희소 특징’ 학습  
- **볼록화 기여**: 비–볼록 모델을 볼록 최적화로 바꾸고, 전역 해 보장  
- **확장 방향**:  
  1. **스펙트럴 노름** 일반화  
  2. **이론적 분석** (일반화 오차·희소성 경로)  
  3. **다른 구조적 제약** (비직교, 계층적 등) 적용  

**결론**: Convex Multi-Task Feature Learning은 여러 관련 작업에서 공유되는 희소 특징을 자동 학습하기 위한 강력한 볼록 최적화 프레임워크를 제시하며, 실험적으로도 효과를 입증했다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c0ce914d-c71f-47f3-a506-08e72194abb0/mtl_feat.pdf
