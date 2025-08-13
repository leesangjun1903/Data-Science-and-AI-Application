# Sequential Bayesian Predictive Synthesis

## 1. 핵심 주장 및 주요 기여  
**Sequential Bayesian Predictive Synthesis** 논문은 다수의 예측분포(agents)들을 결합해 하나의 일관된 시계열 예측분포를 생성하는 동적 베이지안 예측 합성(DBPS) 문제에 대해, 기존의 MCMC 기반 오프라인 방법이 아닌 효율적인 순차 몬테카를로(SMC) 접근을 제안한다.  
- **주요 기여**  
  1. **Rao–Blackwellized Bootstrap Particle Filter**: DLM(Dynamic Linear Model) 합성 함수의 해석적 성질을 활용해, 상태벡터만 파티클로 추적하고 파라미터는 해석적으로 마진화하여 계산 효율을 극대화.  
  2. **MCMC 개입 메커니즘**: 파티클 열화(degeneracy) 시 ESS(Effective Sample Size) 기준으로 적시에 Gibbs 샘플러를 개입시켜 정확도를 보장.  
  3. **할인 계수(discount factors) 적응**: 구조적 변화(예: COVID-19 이후 US 인플레이션 급등)에 빠르게 대응하도록 파티클 필터를 통한 실시간 예측과, 여러 할인 계수에 대한 DBPS 결과를 LDF(Loss Discounting Framework)로 결합해 성능을 개선.  

***

## 2. 해결 과제  
- DBPS 예측분포는 합성함수 αt(yt|xt,Φt)의 적분으로 정의되어 MCMC를 반복 호출해야만 계산 가능.  
- 실시간·순차적 데이터 스트리밍 환경에서는 MCMC 반복이 계산 비용을 폭발적으로 증가시켜 실용적이지 않음.  
- 특히 2020–2022년과 같은 갑작스러운 구조변화를 기존 방식으로는 적시에 반영하기 어려움.

***

## 3. 제안 방법  

### 3.1. 모델 구조  
- K개의 에이전트 예측분포 hk,t(·)를 DLM으로 가정.  
- 합성함수 αt(yt|xt,Φt)도 DLM 형태로 정의:  

$$
    \begin{aligned}
      &y_t = F_t^\top \,\theta_t + \epsilon_t,\quad \epsilon_t\sim N(0,\nu_t),\\
      &\theta_t = \theta_{t-1} + \omega_t,\quad \omega_t\sim N(0,\nu_t W_t),\\
      &\nu_t\text{은 베타-감마 랜덤워크: } \nu_t = \beta\,\gamma_t\,\nu_{t-1},\quad \gamma_t\sim\mathrm{Be}\Bigl(\tfrac{\beta n_{t-1}}{2},\tfrac{(1-\beta)n_{t-1}}{2}\Bigr).
    \end{aligned}
  $$
  
  - $$F_t=(1,\,x_{1t},\dots,x_{Kt})^\top$$, $$\theta_t=(\theta_{0t},\dots,\theta_{Kt})^\top$$.  
  - 할인 계수 $$\beta$$ (잔차 분산), $$\delta$$ (상태 진화 분산) 고정 또는 적응적으로 설정.

### 3.2. SMC 알고리즘  
1. **Rao–Blackwellized Bootstrap Particle Filter**  
   - 파티클 $$\{x_{1:t}^{(i)},W_t^{(i)}\}_{i=1}^M$$만 유지.  
   - 상태 사전분포 제안 $$q_t(x_t)=h_t(x_t)$$ 사용 → 중요도 가중치 $$w_t^{(i)}=p(y_t\mid x_{1:t}^{(i)})$$ 계산만으로 충분.  
   - 파라미터 $$\theta_t,\nu_t$$는 DLM 해석적 업데이트로 마진화.

2. **MCMC 개입 (Intervention)**  
   - ESS $$=\bigl(\sum_i(W_t^{(i)})^2\bigr)^{-1}$$가 임계치 $$C$$ 미만일 때, 해당 시점 t에 Gibbs 샘플러(FFBS) 수행.  
   - MCMC 체인 크기 $$N$$으로 오프라인 후방표본을 다시 초기 파티클로 환원.

3. **할인 계수 적응을 위한 LDF**  
   - 여러 $$(\beta_j,\delta_j)$$ 조합으로 DBPS 예측분포  평가 → LDPL(log-discounted predictive likelihood)  
   - LDF softmax 또는 argmax 가중합으로 실시간 최적 할인 계수를 결합  

***

## 4. 성능 향상  
- **계산 시간**: MCMC 반복 시 전체 과거자료 처리 → 계산 시간이 $$O(t)$$로 선형 증가. 반면 SMC는 $$O(1)$$로 일정(평균 ≈0.56초) [그림1].  
- **정확도**: MCMC 개입 없는 순수 SMC는 파티클 열화로 예측밀도 편향 발생. 개입 시 MCMC 결과와 거의 일치하는 예측 및 파라미터 포스터리어 확보 [그림3,4].  
- **적응력**: 2020–2022년 급등기에도 LDF-DBPS는 폭넓은 예측분산을 반영해 성능 유지, LPDR 개선 [그림5,6].

***

## 5. 한계 및 향후 과제  
- **파티클 열화 임계치 선정**: ESS 임계치 $$C$$ 설정은 경험적·문제별 최적화 필요(표 S1).  
- **대규모 에이전트(K≫)**: 많은 모델 동시 합성 시 변수 선택·차원 저감 연구 필요.  
- **합성 함수 다양화**: 비선형·혼합 합성(mixture synthesis) 등 다른 αt에 대한 맞춤형 SMC 개발 과제.

***

**결론**: 본 논문은 DBPS의 실시간 순차 적용을 위해 SMC와 MCMC의 하이브리드 알고리즘을 제안함으로써, 빠른 계산과 높은 정확도를 동시에 달성하고, 구조적 변화에 민감하게 대응할 수 있는 할인 계수 적응 기법을 도입했다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7c55ee66-862c-42f1-95b8-b4e027da3786/2308.15910v1.pdf
