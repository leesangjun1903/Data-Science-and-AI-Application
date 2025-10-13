# Provably Robust Boosted Decision Stumps and Trees against Adversarial Attacks

**주요 주장 및 기여**  
이 논문은 부스팅된 결정 스텀프(decision stump)와 결정 트리(decision tree)에 대해 **l∞ 공격에 대한 정확한 최악-최선(min–max) 견고 손실 계산과 효율적 최적화 방법**을 제안한다. 특히, 부스팅된 스텀프에 대해 입력당 $$O(T\log T)$$ 시간에 견고 손실을 정확히 계산하고, 전체 앙상블 업데이트를 $$O(n^2T\log T)$$에 수행할 수 있음을 보인다. 또한 부스팅된 트리에 대해서는 견고 손실의 상계(upper bound)를 효율적으로 계산·최적화하여 MNIST, FMNIST, CIFAR-10에서 최첨단의 견고 테스트 오류를 달성한다.[1]

## 1. 해결하고자 하는 문제  
딥 뉴럴 네트워크에 비해 부스팅 트리는 실무에서 높은 정확도, 해석 가능성, 효율성을 인정받지만, **적대적 공격(adversarial attack)에 대한 이론적·계산적 연구는 거의 전무**하다. 특히 l∞ 공격 내에서 앙상블 분류기의 최악-최선 견고 손실(robust loss) 계산과 그에 기반한 학습 절차가 부재했다. 이 논문은 다음 문제를 다룬다.
- 부스팅된 결정 스텀프(Stumps)에서 최적의 적대적 예(examples)에 대응한 정확한 견고 손실 및 테스트 오류 계산  
- 부스팅된 결정 트리(Trees)에서 계산 불가능한 최소화 문제를 **상계(upper bound)** 로 치환해 효율적 최적화

## 2. 제안하는 방법  
### 2.1 부스팅된 결정 스텀프  
각 데이터 $$x$$에 대해 l∞-공격 반경 $$\epsilon$$ 내 최악 손실을 다음과 같이 정의하고, 이를 정확히 계산한다.  

$$
\tilde{L}(u) = \max_{\tilde{x}\in B_\infty(x,\epsilon)} L\bigl(\tilde{G}(x,y)+u\,q(\tilde{x})\bigr),
$$

여기서 $$T$$는 스텀프 개수, $$q(\cdot)$$는 스텀프의 분류 규칙, $$\tilde{G}(x,y)$$는 이전 앙상블 기여, $$u$$는 새로운 스텀프의 가중치다. 가능한 임계값(threshold)을 $$O(n)$$개로 제한하고, 각 임계값별로 이진 탐색(bisection) 또는 볼록 최적화(convex optimization)를 통해 $$w_l,w_r$$를 찾는다. 전체 계산 복잡도는 입력당 $$O(T\log T)$$, 앙상블 업데이트당 $$O(n^2T\log T)$$이다.[1]

### 2.2 부스팅된 결정 트리  
트리 앙상블의 정확한 inner maximization은 NP-하드이므로, 각 트리별 최악 마진을 계산해 **lower bound** $$\tilde{G}(x,y)$$를 구하고 이를 종합해 **robust error 상계**를 유도한다.  

$$
\min_{\|\delta\|_\infty\le\epsilon} yF(x+\delta)
\;\ge\;\sum_{t=1}^T \min_{\|\delta\|_\infty\le\epsilon} y\,u^{(t)}q^{(t)}(x+\delta)
\;=\;\tilde{G}(x,y).
$$

이를 기반으로 전체 앙상블의 상계 손실을 구해 경사하강법 없이 트리별 leaf weight를 업데이트하며 최적화한다.[1]

## 3. 모델 구조  
- **부스팅된 결정 스텀프**: 약한 학습기(stump) $$f_t(x)=u_t\,q_t(x)$$를 순차적으로 추가  
- **부스팅된 결정 트리**: 각 트리는 $$l$$개의 leaf를 가지며, leaf weight $$u\in\mathbb{R}^l$$를 볼록 최적화로 학습  
- **Shrinkage(학습률 $$\alpha$$)**를 도입해 앙상블에 $$\alpha f$$를 더하며, 견고 손실 상계 역시 볼록성으로 인해 단조 감소를 보장  

## 4. 성능 향상  
제안 기법으로 얻은 **provable robust test error**는 다음과 같다.  
- MNIST ($$\epsilon_\infty=0.3$$): **12.5%**  
- FMNIST ($$\epsilon_\infty=0.1$$): **23.2%**  
- CIFAR-10 ($$\epsilon_\infty=8/255$$): **74.7%**  
이는 동등한 공격 조건에서 **견고 합성곱 신경망**과 대등하거나 더 우수한 수준이다.[1]

## 5. 한계  
- 결정 트리에 대한 견고 손실은 **상계**이며, 일부 경우 실제 최소화 결과와 차이가 있을 수 있음  
- 계산 복잡도는 트리 크기(leaf 수)와 데이터 차원에 선형 비례하므로, **매우 큰 트리**나 **고차원 데이터**에선 비용 부담  
- clean accuracy 저하 가능성: 견고 최적화에 집중하며 일반 학습 정확도가 다소 감소할 수 있음

## 6. 일반화 성능 향상  
제안 기법은 **볼록 견고 손실 상계**를 최소화함으로써, 앙상블의 분류 마진(margin)을 적극적으로 증가시킨다. Boosting 이론상 마진 최대화는 **일반화 성능** 향상으로 이어지며, l∞ ball 내 worst-case margin 보장은 **robust generalization**으로도 연결된다. 특히 shrinkage 매개변수 도입으로 **overfitting**을 방지하면서도 견고성–일반화 균형을 맞춘다.[1]

## 7. 향후 연구에 미치는 영향 및 고려사항  
본 연구는 부스팅 모델의 **이론적 견고성(certifiable robustness)** 연구를 촉진하며, 다음 방향에서 후속 연구가 기대된다.
- l2, l1 등 **다른 노름**에 대한 견고 상계 및 최적화  
- **랜덤 스무딩**과 결합한 하이브리드 인증 기법  
- 실무용 대규모 트리 앙상블(예: LightGBM, CatBoost)에서의 **속도 최적화**  
- 견고 학습 시 **clean accuracy** 저하 최소화 위한 다목적 손실 설계  

추후 연구는 계산 효율성과 일반 학습 성능 유지 사이의 균형을 중점적으로 고려해야 한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/854ba1a8-6817-4b50-904e-06fc9fe227ed/1906.03526v2.pdf)
