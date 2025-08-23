# PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization

## 주요 주장 및 기여  
**PTQ4ViT**는 사전학습된 Vision Transformer(ViT)에 후처리 양자화(Post-Training Quantization)를 적용할 때 발생하는 성능 저하 문제를 해결하기 위해 제안된 방법이다.  
1. **분포 특성 분석**  
   - Self-attention의 softmax 출력은 대부분 0에 몰려 있고 일부 큰 값이 중요하나, 균일 양자화 시 작은 값이 0으로 버려지거나 큰 값이 뭉개져 성능이 저하됨을 확인.  
   - MLP의 GELU 출력은 음수 영역 분포가 좁고 양수 영역 분포가 넓어, 동일 비트폭 균일 양자화로는 양·음 모두를 만족시킬 수 없음.  
2. **Twin Uniform Quantization**  
   - softmax 출력과 GELU 출력을 각각 두 구간(R1, R2)으로 분리해 서로 다른 스케일로 양자화함으로써 중요한 큰 값과 음수 영역을 보존.  
   - k-bit 양자화 시 첫 비트를 구간 플래그로, 나머지 k-1비트로 양자화값을 표현하도록 데이터 포맷을 설계하여 기존 하드웨어에서 효율적 연산 지원.  
3. **Hessian Guided Metric**  
   - 기존 MSE, cosine, Pearson 등 로컬 출력 비교 지표 대신, 레이어 출력 변화가 최종 분류 손실에 미치는 영향을 근사한 헤시안 기반 가중치(출력 기울기 제곱값)로 측정하여 스케일 최적화를 수행.  
4. **효율적 프레임워크**  
   - 전방향·역방향 연산으로 각 레이어 출력과 그라디언트를 사전 계산한 뒤, 후보 스케일을 배치 처리로 평가해 빠른 파라미터 탐색 지원.  
   - ImageNet 상 ViT, DeiT, Swin 등 다양한 모델에서 8-bit 양자화 시 0.5% 미만 정확도 손실, 6-bit 시에도 2% 내외 손실을 달성.  

## 해결 과제 및 제안 방법

### 문제 정의  
- **Post-softmax 분포 불균형**: softmax 출력을 단일 스케일로 양자화하면 작은 값은 0으로 소실, 큰 값은 표현 불충분  
- **Post-GELU 비대칭 분포**: 음수·양수 영역 분포 범위 차이로 균일 스케일 적용 어려움  
- **부정확한 스케일 평가 지표**: 로컬 출력 거리 측정 지표가 최종 분류 성능과 일치하지 않음  

### Twin Uniform Quantization  

수식:  

```math
T_k(x;\Delta_{R1},\Delta_{R2})
=
\begin{cases}
\Psi_{k-1}(x,\Delta_{R1}),& x\in R1,\\
\Psi_{k-1}(x,\Delta_{R2}),& x\in R2,
\end{cases}
```

여기서 $$\Psi_{k-1}$$은 $$k-1$$비트 균일 양자화,  
- softmax: $$R1=[0,2^{k-1}\Delta_{R1}),\;R2=$$, $$\Delta_{R2}=1/2^{k-1}$$[1]
- GELU: $$R1=[-2^{k-1}\Delta_{R1},0],\;R2=[0,2^{k-1}\Delta_{R2}]$$, $$\Delta_{R1}$$ 고정  

스케일 정렬: $$\Delta_{R2}=2^m\Delta_{R1}$$로 설정해 비트 시프트로 효율 연산 보장.

### Hessian Guided Metric  
레이어 $$l$$의 출력 변화 $$\hat O^l - O^l$$가 분류 손실 $$L$$에 미치는 영향 근사:  

$$
\min_{\Delta}
\;(\hat O^l - O^l)^\top
\mathrm{diag}\bigl((\tfrac{\partial L}{\partial O^l_i})^2\bigr)\,(\hat O^l - O^l).
$$  

이는 출력별 손실 민감도(출력 기울기 제곱)로 양자화 파라미터를 최적화.

## 모델 구조 및 워크플로우  
1. **레이어별 전·후방 연산**: 캘리브레이션 이미지로 전방 연산해 $$O^l$$, 역방향 연산해 $$\partial L/\partial O^l$$ 저장  
2. **스케일 후보 생성**: 각 레이어별로 weight/activation 또는 구간별 $$\Delta$$ 후보 그리드  
3. **스케일 탐색**: Twin Uniform Quant. 적용해 $$\hat O^l$$ 계산, Hessian Guided Metric으로 최적 스케일 선택  
4. **양자화 적용**: 전체 레이어에 최적 스케일 부여 후 배포

## 성능 향상 및 한계  
- **향상**: 8-bit 양자화 시 일반적 PTQ 대비 0.5% 이하 손실, 6-bit 시 2% 내외로 성능 유지 (ImageNet 기준). Swin 계열은 0.15% 미만 손실.  
- **한계**:  
  - 4-bit 이하 저비트폭 시 mixed-precision 필요.  
  - 캘리브레이션 이미지 수(32개)·GPU 메모리 소요 고려.  
  - 비전 트랜스포머 외 다른 아키텍처 일반화 검증 필요.

## 일반화 성능 향상 관점  
Twin Uniform Quant.과 Hessian Guided Metric은  
- **희소하지만 중요한 활성화 보존**: self-attention의 결정적 값 유지로 일반화 성능 저하 방지  
- **레이어 민감도 고려**: 손실 기울기에 따른 스케일 최적화로 전이 학습·미세조정 없는 상황에서도 안정적 성능 유지  
따라서 소규모 캘리브레이션 데이터만으로도 다양한 도메인에 대해 강건한 일반화 가능성 제시.

## 향후 연구 방향 및 고려 사항  
- **Mixed-Precision 자동화**: 4-bit 이하 영역에서 모델 크기·성능 균형을 위한 bit-width 배치 최적화  
- **다양한 트랜스포머 변형 검증**: DETR, Swin-V2, MLP-Mixer 등으로 확장 연구  
- **동적 양자화 기법 결합**: 런타임 입력 분포 변화 대응을 위한 online quantization  
- **하드웨어 가속기 통합**: 구간 플래그 기반 연산 최적화를 위한 ASIC/FPGA 지원 방안 검토

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/ce2f6cff-d9da-41a8-97d0-b95c1a361d54/2111.12293v3.pdf)
