# Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations

**주요 주장**  
Physics Informed Neural Networks(PINNs)를 활용해 **산발적이고 소량의 관측 데이터**만으로도 물리 법칙을 기술하는 **비선형 편미분방정식(PDE)** 의 미지 파라미터를 **고정밀도로 추정**하고, 추가적인 경계·초기 조건 없이도 완전한 연속 해(pressure 등)를 복원할 수 있다.

**주요 기여**  
1. **연속 시공간 모델 (Continuous Time PINNs)**  
   -  PDE $$u_t + \mathcal{N}[u;\lambda]=0$$의 잔차항  
   
$$
     f(t,x)=u_t + \mathcal{N}[u;\lambda]
   $$  
   
   을 자동미분으로 계산, 손실함수  
   
$$
     \mathrm{MSE}=\frac1N\sum_i|u(t_i,x_i)-u_i|^2
     +\frac1N\sum_i|f(t_i,x_i)|^2
   $$  
   
   를 최소화하며 $$\lambda$$와 신경망 파라미터를 동시 학습.  
   -  Burgers 방정식과 Navier–Stokes 방정식(원통 유동), KdV 방정식 등을 대상으로 1% 이하 노이즈 환경에서도 $$\lambda$$를 0.01% 내외로 정확 추정하고, 압력장·속도장 전역 복원에 성공[1].  

2. **이산 시공간 모델 (Discrete Time PINNs)**  
   -  다단계 Runge–Kutta(time-stepping) 공식을 신경망 구조에 내장:  
   
$$
     u^{n+c_i}=u^n - \Delta t\sum_j a_{ij}\mathcal{N}[u^{n+c_j};\lambda],\ 
     u^{n+1}=u^n - \Delta t\sum_j b_j\mathcal{N}[u^{n+c_j};\lambda]
   $$  
   
   -  두 시점의 스냅샷 데이터만으로 $$\lambda$$ 추정 가능. 시점 간격 $$\Delta t$$가 매우 커도, 임의로 많은 Runge–Kutta 단계(q)를 사용하여 **대형 타임스텝**을 안정적으로 처리.  
   -  Burgers, KdV 방정식 사례에서 $$\Delta t$$ 최대 0.8까지도 $$\lambda$$ 0.1% 내외 정확도 유지.  

3. **모델 구조**  
   - 연속 모델: 9층·20뉴런(또는 5층·100뉴런) tanh 활성화  
   - 이산 모델: 4층·50–200뉴런, 출력층 뉴런 수 = q (Runge–Kutta 단계 수)  
   - 자동미분 활용으로 압력·추가 변수 복원  

4. **성능 향상 및 한계**  
   - **강력한 정규화**: PDE 잔차항이 네트워크의 과적합 억제  
   - **노이즈·희소 데이터**에도 $$\lambda$$와 전역 해 안정 복원  
   - **한계**:  
     - 불확실성 정량화 부재 (신뢰도 구간 미제공)  
     - 고차원(3D+)에서 계산량·collocation 점 폭발적 증가  
     - 전역 최적화 보장 없음  

## 일반화 성능 향상 가능성  
PINNs는 **물리 법칙**을 학습 제약으로 직접 통합하여  
- 데이터 희소·잡음 상황에서도 모델이 물리적으로 일관된 해 공간만 탐색  
- 학습 시 unseen 영역에서도 물리 일관성 유지 → **강건한 일반화**  
- 이산 모델의 다단계 Runge–Kutta 활용으로 시공간 간 불연속·이방성 문제에도 확장 가능  

이로써 양질의 수치해석 데이터가 부족한 물리 실험·관측 분야에서 **일반화 성능**이 획기적으로 향상될 잠재력 보유한다.

## 향후 연구 영향 및 고려사항  
- **불확실성 정량화**: Bayesian PINNs 등으로 신뢰도 평가 추가  
- **고차원 확장**: 적응형 collocation, 중요도 샘플링으로 효율 개선  
- **멀티피직스 통합**: 서로 다른 PDE를 하나의 PINN에 결합  
- **하드웨어 최적화**: 자동미분 비용 절감, 분산 처리  
- **수치해석+ML 협업**: 고전적 수치기법(Runge–Kutta, 유한요소 등)과의 시너지 추구  

PINNs Part II는 “물리 법칙 추정”의 **새 패러다임**을 제시하며, 과학·공학 분야에서 **데이터 효율적 모델 추정**과 **연속 파라미터 복원**을 위한 토대를 마련한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/93fd92da-0957-47a5-b25a-34e325e32608/1711.10566v1.pdf
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/1ecd12d3-8dbc-4b7e-b672-652f675185b4/1711.10561v1.pdf
