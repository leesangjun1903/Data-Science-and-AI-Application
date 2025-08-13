# Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations

## 주요 주장  
**Physics Informed Neural Networks (PINNs)**는 물리 법칙(비선형 편미분방정식)으로부터 도출된 제약조건을 학습 과정에 통합함으로써,  
– 데이터가 희소한 상황에서도 안정적 학습 및 높은 예측 정확도를 달성  
– 전통적 수치해석 기법과 달리 도메인 이산화 없이 “mesh-free” 방식으로 해를 직접 근사  
– 자동미분을 활용해 네트워크가 PDE 잔차를 최소화하도록 구성  

## 주요 기여  
1. **연속 시공간 모델 (Continuous Time Models)**  
   - 물리 법칙 $$u_t + \mathcal{N}[u]=0$$을 잔차항 $$f(t,x)=u_t+\mathcal{N}[u]$$으로 정의  
   - 손실함수에 초기·경계 데이터 항과 PDE 잔차 항을 결합:  

  $$
       \text{MSE} = \frac1{N_u}\sum_i|u(t_i^u,x_i^u)-u_i|^2
       +\frac1{N_f}\sum_j|f(t_j^f,x_j^f)|^2
     $$
   
   - Burgers 방정식, Schrödinger 방정식 예제에서 데이터 수백 점, collocation 점 수만 점만으로도 상대 L₂ 오차 $$10^{-3}$$ 이하 달성  

2. **이산 시공간 모델 (Discrete Time Models)**  
   - 일반 q단계 Runge–Kutta(time-stepping) 규칙을 네트워크에 내장  
   - 한 번의 대형 타임스텝(예: $$\Delta t=0.8$$, q=500)으로 시공간 전체 해 예측  
   - 손실함수:  

```math
\text{SSE} = \sum_{j=1}^{q+1}\sum_{i=1}^{N_n}\bigl|u^{n}_j(x_{n,i})-u_{n,i}\bigr|^2 +\text{경계조건 손실}
```
   
   - Burgers 방정식에서 단일 스텝 예측 시 상대 L₂ 오차 $$8.2\times10^{-4}$$  

3. **모델 구조**  
   - **연속 모델**: 은닉 9층·20뉴런(또는 5층·100뉴런) tanh 활성화  
   - **이산 모델**: 은닉 4층·50–200뉴런 tanh 활성화, 출력층 뉴런 수 = q+1  
   - 자동미분으로 네트워크 출력에 대한 시간/공간 미분 계산  

4. **성능 향상 및 일반화**  
   - PDE 잔차 항이 **강력한 정규화** 역할을 하여 과적합 억제  
   - 작은 Nu(수백 점)에서도 높은 정확도 달성  
   - 네트워크 용량(층·뉴런 수)·collocation 점 수(Nf)에 비례해 오차 감소  
   - 이산 모델에서 고차 Runge–Kutta 단계(q) 증가 → 대형 Δt에서도 안정적 예측  

5. **한계**  
   - **불확실성 정량화** 부재: Gaussian Process 기반 방법처럼 예측 신뢰도 제공 어려움  
   - 고차원 공간(3D 이상)에서 collocation 점 수 폭발적 증가  
   - 최적화가 전역 수렴 보장 없음(L-BFGS, SGD 등)  

## 일반화 성능 향상 관련  
PINNs의 **물리 법칙 통합**은 학습 가능한 해 공간을 좁혀,  
– 노이즈·희소 데이터에도 과적합 억제  
– PDE 잔차가 미니멈이 되도록 학습 → 보이지 않는 영역에서도 물리 일관성 유지  
– 네트워크 크기(층·뉴런) 확장 시도 시에도 안정적 학습  

이러한 특성은 특히 데이터가 한정된 실험·관측 분야에서 **강건한 일반화** 능력을 제공  

## 향후 연구 영향 및 고려사항  
- **데이터 효율적 수치 해석기법**: PINNs는 mesh-free 방식으로 복합 영역·비정형 경계 문제에 적용 가능  
- **멀티피직스·다중스케일 모델링**: 서로 다른 PDE를 한 네트워크에 통합해 상호작용 캡처  
- **불확실성 정량화** 및 **베이지안 PINNs** 연구 필요  
- **고차원 문제 확장**: collocation 점 감소 기법(importance sampling, adaptive refinement)  
- **하드웨어 가속** 최적화: 자동미분 비용 절감 및 대규모 네트워크 학습  

이 논문은 물리 법칙과 딥러닝 결합의 새로운 패러다임을 제시하며, 과학·공학 계산에 혁신적 전환을 예고한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/1ecd12d3-8dbc-4b7e-b672-652f675185b4/1711.10561v1.pdf
