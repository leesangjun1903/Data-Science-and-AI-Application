# Optuna: A Next-generation Hyperparameter Optimization Framework

**핵심 주장 및 주요 기여**  
Optuna는 기존 하이퍼파라미터 최적화 도구들이 가진  
1) 정적(define-and-run) 검색 공간 정의의 한계  
2) 비효율적 프루닝(pruning) 전략  
3) 확장성과 배포 용이성 부족  
문제를 해결하기 위해 다음 세 가지 설계 원칙을 제안한다.  
- **Define-by-Run API**: 사용자 코드 실행 중에 동적으로 탐색 공간을 구성  
- **효율적 샘플링·프루닝 기법**: TPE, CMA-ES 등 다양한 기법과 ASHA(Asynchronous Successive Halving)를 적용  
- **경량·확장 가능한 아키텍처**: 단일 커맨드 설치, 인메모리 및 RDB 백엔드, 대화형 및 분산 환경 지원  

이로써 Optuna는 복잡한 조건부 탐색 공간을 간결한 코드로 모델링하고, 불필요한 실험을 조기 종료하여 자원을 절감하며, 단일 노트북부터 대규모 쿠버네티스 클러스터까지 유연하게 배포할 수 있음을 보여준다.

***

## 1. 문제 정의  
기존 하이퍼파라미터 최적화 프레임워크들은  
- 탐색 공간을 사전에 전부 정의해야 하는 **정적 API**  
- 런타임 자원 활용을 최적화하지 못하는 **비효율적 프루닝**  
- 설치·배포 복잡성으로 인한 **확장성 제약**  
등의 한계를 지닌다.  
이로 인해 대규모 조건부 변수, 동적 네트워크 구조를 다루기 어렵고, 불필요한 실험이 리소스를 낭비하며, 다양한 환경에 배포하기가 번거롭다.

***

## 2. 제안 방법 및 모델 구조  

### 2.1 Define-by-Run API  
사용자는 Objective 함수 내부에서 `trial.suggest_*` 호출로 탐색 공간을 *동적으로* 구성한다.  
```python
def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers = [trial.suggest_int(f"n_units_{i}", 1, 128) for i in range(n_layers)]
    model = MLP(tuple(layers))
    ...
```
- 루프와 조건문으로 복합적·이종(hierarchical) 변수 공간을 직관적으로 표현  
- 배포 시 `FixedTrial` 객체로 최적 파라미터를 고정하여 재사용  

### 2.2 효율적 샘플링  
- **독립 샘플링**: TPE(Tree-structured Parzen Estimator)  
- **관계 샘플링**: CMA-ES, GP-BO 등  
동적 탐색 공간에서도 변수간 상호관계를 후속 학습으로 추론해 적용  

### 2.3 ASHA 기반 프루닝  
Asynchronous Successive Halving Algorithm(ASHA)을 변형하여,  
1) 단계별 중간 평가값을 `trial.report(value, step)`로 기록  
2) `trial.should_prune(step)`로 상위 η분위 밖 성능을 조기 종료  

$$ \text{rung} = \max\bigl(0, \log_\eta(\lfloor\tfrac{\text{step}}{r}\rfloor)-s \bigr) $$  

$$\text{prune if } v_{\text{trial}} < \text{top}\_{\lfloor N/\eta\rfloor}\bigl(v_{\text{all}}\bigr) $$  

- 분산 환경에서 워커 동기화 없이 병렬 확장성 확보  

### 2.4 시스템 아키텍처  
- **경량 모드**: 인메모리 스토리지, Jupyter 노트북  
- **분산 모드**: SQLite 또는 RDB 백엔드, 다중 프로세스·노드  
- 실시간 대시보드로 실험 상태·파라미터 시각화 지원  

***

## 3. 성능 향상 및 한계  

### 3.1 벤치마크 성능  
- **TPE+CMA-ES** 조합: 56개 블랙박스 최적화 테스트에서 대부분 라이벌을 통계적으로 우월하거나 동등하게 성능 발휘  
- **실험 시간**: GPyOpt 대비 10배 이상 빠른 Trial당 계산 시간  
- **프루닝 효과**: AlexNet on SVHN 실험에서 무프루닝 대비 35→1,278 trials 증가, 테스트 오차 조기 개선  

### 3.2 분산 확장성  
- 1→8 워커에서 최적화 속도가 선형 증가  
- 워커 수에 무관하게 Trial당 성능 일관 유지  

### 3.3 실제 적용 사례  
- **Kaggle 대회**: Open Images Object Detection 2위  
- **TOP500 HPL**: 슈퍼컴퓨팅 Linpack 튜닝  
- **RocksDB**: 372s → 30s 성능 향상  
- **FFmpeg 인코더 튜닝**: Preset 수준의 복원 성능 달성  

### 3.4 한계  
- **GP-BO 대체성능**: 일부 테스트에서 GP-BO에 뒤처짐  
- **대규모 조건부 공간**: 매우 복잡한 논리적 분기 시 메모리·계산 오버헤드 가능성  
- **사용자맞춤 샘플러**: 외부 샘플러 통합 시 인터페이스 추가 개발 필요  

***

## 4. 일반화 성능 향상 가능성  
- **동적 구성**이 복합 모델·하이브리드 구조 탐색을 용이하게 하여 과적합 방지  
- ASHA 프루닝이 중간 학습 곡선을 활용, 불필요한 과대적합 실험 조기 취소  
- 다중 워커와 통합된 변수‐성능 관계 학습으로 전역 최적화 강화  
→ 다양한 모델에 적용 시 **검증 셋 일반화 오차 감소** 및 **재현성** 보장 가능  

***

## 5. 향후 연구 영향 및 고려 사항  

1. **AutoML 통합**: Optuna의 define-by-run 개념을 AutoML 파이프라인에 확장 적용  
2. **하이브리드 샘플러 개발**: 동적 관계 추론을 심화한 Bayesian Neural Network 기반 샘플링  
3. **메타러닝 결합**: 이전 스터디 기록 활용한 메타 최적화로 초기 탐색 가속  
4. **대규모 조건부 공간**: 메모리·계산 절약형 데이터 구조 및 지연 평가 기법 고도화  
5. **이해가능성 강화**: 탐색 경로·프루닝 의사결정 시각화로 사용자 신뢰도 제고  

이러한 고려를 통해 Optuna의 설계 철학은 차세대 최적화 프레임워크와 AutoML 연구 전반에 폭넓게 기여할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7f79c31b-986b-4b26-be8b-3f67e56c706d/1907.10902v1.pdf)
