# Model Lightweight 가이드

핵심 요약: 이 글은 제한된 GPU 메모리와 지연 시간 제약 속에서 대형 NLP 모델을 빠르고 가볍게 배포하는 실전 가이드를 제공합니다. 특히 **Quantization**, **Pruning**, **Knowledge Distillation**의 원리와 장단점을 설명하고, PyTorch/Hugging Face 기반의 예시 코드를 통해 연구 수준의 방법을 실무에 적용하는 방법을 제시합니다.[1][2][3]

## 왜 모델 경량화인가
대형 Transformer는 추론에서도 메모리와 지연 시간이 병목이 됩니다. 서비스 SLA를 만족하려면 파라미터 수와 연산량을 줄여 지연을 낮추고, 메모리 footprint(환경적·에너지적 영향)를 줄여 더 저렴한 하드웨어(GPU T4 등)에서 안정적으로 서빙해야 합니다. 경량화는 정확도를 크게 잃지 않으면서 모델을 작고 빠르게 만들어 배포 효율을 높이는 실전적 해법입니다.[2][3][1]

## 세 가지 축: Quantization, Pruning, Distillation
- Quantization: 가중치/활성화를 INT8/INT4 등 저정밀로 내려 모델 크기와 메모리 대역폭을 줄입니다. PTQ는 빠르지만 과도한 비트 폭 축소 시 정확도 저하가 있고, QAT는 훈련 중 양자화를 모사해 정확도 회복을 돕습니다. PyTorch 최신 QAT 흐름은 PTQ 대비 Llama 계열에서 정확도 저하의 최대 96%를 회복하는 사례가 보고되었습니다.[4][3]
- Pruning: 중요도가 낮은 가중치/채널/레이어를 제거해 FLOPs와 파라미터를 줄입니다. 구조적(채널/헤드/레이어)과 비구조적(스파스) 접근이 있으며, 재학습과 zero-skipping 하드웨어 최적화로 추론 가속이 가능합니다.[5][2]
- Knowledge Distillation: 큰 Teacher의 지식을 작은 Student로 전수합니다. DistilBERT는 BERT 대비 40% 축소, 60% 속도 향상, 97% 언어 이해력 유지 결과를 보였습니다. 다중 타깃을 결합한 고급 증류는 더 큰 속도 향상을 보이기도 합니다.[6][7][2]

## 언제 무엇을 쓸까
- 메모리 부족이 심각하고 재학습 여건이 제한된다면: PTQ→필요 시 소량의 QAT 파인튜닝으로 보정합니다.[3][4]
- 모델 최적화에 있어서 지연시간(latency)이 성능에 결정적 역할을 하여 이 지연을 줄이거나 관리하는 것이 중요할 때, GPU가 스파스/INT8 가속 지원된다면: 구조적 Pruning과 INT8 Quantization을 결합합니다.[1][5]
- 모델 사이즈를 크게 줄이며 정확도 유지가 중요하다면: Teacher-Student Distillation을 적용하고, 필요 시 Pruning/Quantization과 함께 파이프라인화합니다.[6][1]

## 설계 원칙과 트레이드오프
- 정확도-크기-속도 균형: 동일 파라미터 축소라도 Task/데이터에 따라 정확도 영향이 다릅니다. GLUE, MMLU 등 벤치마크 데이터셋으로 각 단계마다 확인하세요.[2][3]
- 보안/강건성 고려: 일부 연구는 압축이 적대적 취약성에 영향을 준다고 보고합니다. 보안 민감 환경에선 강건성 평가를 병행해야 합니다.[8][9]
- 조합 최적화: 양자화+프루닝+증류를 공동 최적화하는 JPQD 파이프라인처럼 통합 접근이 실무 복잡도를 낮추고 효율/정확도를 동시에 달성합니다.[1]

### JPQD, Joint Pruning, Quantization and Distillation
JPQD는 OpenVINO의 신경망 압축 프레임워크(NNCF)에서 개발한 Joint Pruning, Quantization and Distillation의 약자로, 사전학습된 트랜스포머 모델의 추론 성능을 향상시키기 위해 가지치기(pruning), 양자화(quantization), 지식 증류(distillation)를 병렬로 수행하는 단일 최적화 파이프라인입니다. 이를 통해 모델 크기를 크게 압축하고 실행 성능을 크게 개선할 수 있습니다.

특히 BERT-base 모델에 적용 시 5.24배의 압축률과 4.19배의 실행 성능 향상을 달성하였으며, 인텔 플랫폼에서 최적화된 OpenVINO 런타임으로 바로 배포할 수 있는 구조화된 가지치기 및 양자화된 모델을 출력합니다.

## 실전 레시피: BERT/Transformer 기준
- 목표: VRAM 23GB→16GB로, 지연 30% 절감, 정확도 손실 <1%를 가정합니다.[3][1]
- 단계:
  1) 기준선 측정: 파라미터 수, VRAM 피크, QPS/Latency, 검증 정확도.[2][3]
  2) PTQ 시도: INT8 가중치, 동적/정적 활성화 스케일 탐색.[4][3]
  3) 정확도 회복: QAT로 3~5 epoch 미세 조정, 낮은 LR 사용.[4][3]
  4) 구조적 Pruning: 헤드/중간 차원/레이어 축소 후 미세 조정.[5][2]
  5) Distillation: Teacher 로짓/중간표현/어텐션 지도 다중 손실 결합.[6][2]
  6) 통합 최적화: JPQD/프레임워크를 활용한 공동 파이프라인.[1]

## 코드 예시 1: Hugging Face + PyTorch PTQ→QAT(텍스트 분류)
- 개요: 사전학습 BERT-base를 로드하고, 먼저 PTQ로 INT8 가중치 적용 후 정확도 확인, 필요 시 QAT로 소량 파인튜닝합니다.[3][4]

```python
# 설치: pip install transformers datasets evaluate accelerate torch torchvision torchaudio
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch

model_id = "bert-base-uncased"
num_labels = 2
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)

dataset = load_dataset("glue", "sst2")
def tokenize(ex):
    return tokenizer(ex["sentence"], truncation=True, padding="max_length", max_length=128)
dataset = dataset.map(tokenize, batched=True).rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

# 1) 기준선 평가
from evaluate import load as load_metric
metric = load_metric("glue", "sst2")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return metric.compute(predictions=preds, references=labels)

args = TrainingArguments(output_dir="out", per_device_eval_batch_size=64)
trainer = Trainer(model=model, args=args, eval_dataset=dataset["validation"], tokenizer=tokenizer, compute_metrics=compute_metrics)
base_metrics = trainer.evaluate()

# 2) 간단 PTQ: Weight-only INT8 (예: torch.compile 이후 weight quant; 실제 배포는 onnx/openvino/execuTorch 권장)
# 실무에서는 torchao/onnxruntime/openvino 백엔드 INT8 경로를 사용
try:
    import torchao.quantization as aoq  # PyTorch QAT/Quant 툴킷(예시)
    model_int8 = aoq.quantize_(model, weights_only=True, dtype=torch.int8)
except Exception:
    model_int8 = model  # 대체 경로

ptq_metrics = Trainer(model=model_int8, args=args, eval_dataset=dataset["validation"], tokenizer=tokenizer, compute_metrics=compute_metrics).evaluate()

# 3) QAT로 소량 파인튜닝(정확도 회복)
# 주의: 실제 QAT는 fake-quant 모듈 삽입 후 낮은 LR로 수 epoch 학습
finetune_args = TrainingArguments(
    output_dir="out_qat",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=1e-5,  # 작은 LR
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_steps=50,
)
trainer_qat = Trainer(model=model_int8, args=finetune_args, train_dataset=dataset["train"].shuffle(seed=42).select(range(5000)),
                      eval_dataset=dataset["validation"], tokenizer=tokenizer, compute_metrics=compute_metrics)
trainer_qat.train()
qat_metrics = trainer_qat.evaluate()

print("Baseline:", base_metrics)
print("PTQ:", ptq_metrics)
print("QAT:", qat_metrics)
```
이 스니펫은 연구 코드 수준의 흐름을 보여줍니다. 실제 프로덕션에서는 torchao/torchtune 기반 QAT, XNNPACK/execuTorch로 온디바이스 경량화를 적용하거나 OpenVINO IR로 내리는 경로를 권장합니다.[4][3]

## 코드 예시 2: 구조적 Pruning + Distillation(Transformer 인코더)
- 개요: Teacher=BERT-base, Student=작은 층 수/히든 차원으로 정의. 헤드/레이어를 줄인 Student에 다중 손실(로짓, 중간 히든, 어텐션)을 적용합니다.[6][2]

```python
import torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer

teacher_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(teacher_id)
teacher = AutoModel.from_pretrained(teacher_id, output_hidden_states=True, output_attentions=True).eval()

# 간단 Student: 레이어 수 축소(예: 6 layers)
from transformers import BertConfig, BertModel
tcfg = teacher.config
scfg = BertConfig(
    vocab_size=tcfg.vocab_size,
    hidden_size=384,          # 축소
    num_hidden_layers=6,      # 축소
    num_attention_heads=6,    # 축소(헤드 프루닝 효과)
    intermediate_size=1536,   # 축소
)
student = BertModel(scfg, add_pooling_layer=False)
proj_hidden = nn.Linear(scfg.hidden_size, tcfg.hidden_size)  # feature alignment

kl = nn.KLDivLoss(reduction="batchmean")
mse = nn.MSELoss()

def kd_loss(batch):
    with torch.no_grad():
        t_out = teacher(**batch)
    s_out = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    # 로짓/feature가 없다면 Downstream 헤드 추가 후 distill 권장
    # 여기서는 중간표현/어텐션 일치 손실 예시
    hid_loss = 0.0
    for s_h, t_h in zip(s_out.hidden_states, t_out.hidden_states[-len(s_out.hidden_states):]):
        s_up = proj_hidden(s_h)
        hid_loss = hid_loss + mse(s_up, t_h)

    att_loss = 0.0
    for s_a, t_a in zip(s_out.attentions, t_out.attentions[-len(s_out.attentions):]):
        att_loss = att_loss + mse(s_a, t_a)

    return hid_loss + 0.1 * att_loss

# Pruning 팁: 특정 레이어/헤드 중요도 기준으로 제거 후 위 loss로 미세 조정
# 실제 구조적 프루닝/검색은 AutoCompress 또는 JPQD류 프레임워크 사용 권장
```
증류는 DistilBERT처럼 사전학습 단계에서 적용하면 더 큰 이득을 볼 수 있고, 과제별 미세 조정에서 다중 타깃을 결합하면 정확도 손실을 최소화할 수 있습니다.[2][6]

## 프레임워크 · 툴킷
- OpenVINO NNCF: QAT/PTQ, JPQD로 트랜스포머를 공동 최적화해 IR로 내리고 Intel 런타임에서 즉시 배포합니다.[1][4]
- PyTorch torchao/torchtune: LLM 대상 QAT 파이프라인을 제공하며 PTQ 대비 정확도 저하를 크게 회복합니다.[3]
- NNI Compression: 다양한 Pruning/Quantization 알고리즘을 손쉽게 적용하는 학습 루프 템플릿을 제공합니다.[10]

## 고급 팁
- 다중 증류 타깃: 로짓+히든+어텐션을 함께 학습하면 더 작은 Student에서도 정확도 손실이 줄어듭니다.[6][2]
- 구조 축소 레시피: 깊이(레이어), 너비(히든/FFN), 어텐션 헤드를 함께 탐색하고, 소량 데이터+증류로 재학습하면 동일 계열 모델을 2~4배 압축하면서 경쟁력 유지가 가능합니다.[11][2]
- 동적 가속: 초기 레이어의 어텐션 범위를 제한하는 등 동적 추론 가속으로 지연을 추가로 줄일 수 있습니다.[2]

## 실험 체크리스트
- 벤치마크: GLUE/MNLI, MMLU, HellaSwag, WikiText 등에서 기준선→PTQ→QAT→Pruning→Distill 단계별 비교를 기록합니다.[3][2]
- 시스템 지표: 피크 VRAM, 모델 크기(MB), QPS, p95/p99 latency, 에너지 사용 등을 함께 보고합니다.[1][3]
- 강건성/보안: 적대적 공격/분포 이동에 대한 민감도 변화를 병행 평가합니다.[9][8]

## 참고 자료
- Joint Pruning+Quantization+Distillation 파이프라인(Optimum-Intel/JPQD) 개요와 실전 가이드.[1]
- PyTorch LLM QAT 엔드투엔드 튜토리얼 및 코드 흐름, 정확도 회복 사례.[3]
- BERT 압축 연구 리뷰: 다중 증류 타깃, 행렬 분해, 동적 가속 등 전략 비교.[2]
- DistilBERT 원 논문: 40% 축소, 60% 속도 향상, 97% 성능 유지.[7][6]
- Pruning/Distillation 강의 슬라이드로 개념 정리.[5]
- NNI Compression QuickStart로 프루너/양자화기 적용 패턴.[10]
- LLM/코드 모델 압축과 강건성에 관한 최신 실증 연구 동향.[8][9]

이 가이드를 바탕으로, 대상 태스크와 배포 하드웨어에 맞는 조합(예: INT8 QAT + 구조적 Pruning + 다중 타깃 Distillation)을 설계하고, 단계별로 정확도-지연-메모리 추이를 계측하면서 목표 SLA를 만족하도록 수렴시키면 됩니다.[2][3][1]

[1](https://blog.openvino.ai/blog-posts/joint-pruning-quantization-and-distillation-for-efficient-inference-of-transformers)
[2](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00413/107387/Compressing-Large-Scale-Transformer-Based-Models-A)
[3](https://pytorch.org/blog/quantization-aware-training/)
[4](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/compressing-models-during-training/quantization-aware-training-pytorch.html)
[5](https://harvard-iacs.github.io/2020F-AC295/lectures/lecture9/presentation/lecture9.pdf)
[6](https://arxiv.org/abs/1910.01108)
[7](https://arxiv.org/pdf/1910.01108.pdf)
[8](https://www.semanticscholar.org/paper/20cd4344bc4a426bbc0b71e237be1cf6f15be13f)
[9](https://ieeexplore.ieee.org/document/10992473/)
[10](https://nni.readthedocs.io/en/v1.9/Compression/QuickStart.html)
[11](https://arxiv.org/abs/2407.14679)
[12](https://moon-walker.medium.com/ai-service%EB%A5%BC-%EC%9C%84%ED%95%9C-%ED%98%84%EC%8B%A4%EC%A0%81%EC%9D%B8-%EC%A0%91%EA%B7%BC-%EB%B0%A9%EB%B2%95-2-model-lightweight-588c0d650af2)
[13](https://ieeexplore.ieee.org/document/10163237/)
[14](https://dl.acm.org/doi/10.1145/3643488.3660293)
[15](https://ieeexplore.ieee.org/document/11126680/)
[16](https://arxiv.org/abs/2303.17612)
[17](https://www.ijcesen.com/index.php/ijcesen/article/view/944)
[18](https://arxiv.org/abs/2409.00592)
[19](https://ieeexplore.ieee.org/document/10911679/)
[20](https://aclanthology.org/2022.acl-long.107.pdf)
[21](http://arxiv.org/pdf/1510.00149.pdf)
[22](http://arxiv.org/pdf/2410.17170.pdf)
[23](http://arxiv.org/pdf/2205.11141v1.pdf)
[24](http://arxiv.org/pdf/2208.11580v2.pdf)
[25](https://arxiv.org/html/2412.13737)
[26](https://arxiv.org/pdf/2303.03106.pdf)
[27](https://arxiv.org/pdf/2106.14681.pdf)
[28](http://arxiv.org/pdf/2412.16719.pdf)
[29](https://arxiv.org/pdf/2203.11239.pdf)
[30](https://github.com/cedrickchee/awesome-ml-model-compression)
[31](https://www.reddit.com/r/learnmachinelearning/comments/132wft5/links_good_reads_about_model_compression/)
[32](https://www.linkedin.com/pulse/model-compression-techniques-quantization-pruning-amit-kharche-wtumf)
[33](https://www.youtube.com/watch?v=FLkUOkeMd5M)
[34](https://www.ijcai.org/proceedings/2020/0341.pdf)
[35](https://wandb.ai/byyoung3/Generative-AI/reports/Quantization-Aware-Training-QAT-A-step-by-step-guide-with-PyTorch--VmlldzoxMTk2NTY2Mw)
[36](https://towardsdatascience.com/model-compression-make-your-machine-learning-models-lighter-and-faster/)
[37](https://tutorials.pytorch.kr/prototype/pt2e_quant_qat.html)
[38](https://arxiv.org/html/2305.09098v2)
[39](https://mlops.substack.com/p/quantization-aware-training-in-pytorch)
[40](https://aclanthology.org/2021.emnlp-main.832.pdf)
[41](https://github.com/Qualcomm-AI-research/transformer-quantization)

https://moon-walker.medium.com/ai-service%EB%A5%BC-%EC%9C%84%ED%95%9C-%ED%98%84%EC%8B%A4%EC%A0%81%EC%9D%B8-%EC%A0%91%EA%B7%BC-%EB%B0%A9%EB%B2%95-2-model-lightweight-588c0d650af2
