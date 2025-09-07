# Foundation Model 가이드
레이블이 지정되지 않은 광범위한 데이터 집합에 대해 훈련된 대규모 인공 지능 모델로, 광범위한 다운스트림 작업에 적용할 수 있는 AI 모델 유형을 말합니다.

대규모 머신 러닝(ML) 모델은 방대한 양의 데이터를 대규모로 훈련(self-supervised learning, semi-supervised learning) 다양한 다운스트림 작업에 적용할 수 있는 모델을 의미합니다.

다음 글은 Foundation Model을 처음 접하는 딥러닝 전공 대학생을 위해, 개념 이해부터 실제 코드 예제까지 한 번에 익히도록 구성한 가이드입니다. 핵심은 “넓게(pre-train), 필요에 맞게(adapt), 안전하게(align)”입니다.[1][2][3]

## 한눈에 핵심
Foundation Model은 광범위한 데이터로 사전학습된 대규모 신경망으로, 다양한 다운스트림 작업에 쉽게 **적응**하도록 만들어진 범용 **기반** 모델입니다.[2][3]
대학 연구·프로덕션 모두에서 “사전학습 모델 활용 + 경량 미세조정”이 비용·시간 절감의 표준 워크플로우입니다.[3][1]

## Foundation Model이란
스탠퍼드 HAI/CRFM은 Foundation Model을 “넓은 데이터로 대규모 자기지도 방식으로 학습되어, 다양한 과제에 적응 가능한 모델”로 정의합니다. 이는 언어·비전·멀티모달·코드 등 여러 영역에 적용됩니다.[2]
IBM은 이러한 모델이 엔터프라이즈에서도 높은 정확도와 비용 효율을 보이며, 적은 라벨로도 새로운 언어·태스크에 빠르게 확장된다고 설명합니다.[3]

## 왜 지금 중요한가
NVIDIA는 트랜스포머·확산 등 범용 아키텍처가 대규모 데이터로 학습되며 “출현(창발)·균질화(표준화)”라는 현상과 함께 산업 전반 활용이 급증했다고 정리합니다.[1]
이제 하나의 기반 모델을 가져와 도메인 데이터로 경량 적응하면, 번역·검색·헬스케어·로보틱스 등으로 빠르게 확장 가능합니다.[1]

## 장단점 요약
- 장점: 라벨링 비용 절감, 빠른 배포, 다양한 태스크에 범용 적용, 미세조정으로 성능 향상 여지.[3][1]
- 단점/과제: 편향·환각·저작권·운영비용·안전성 관리가 필요하며, 거대 모델 학습은 고비용·고컴퓨팅 요구.[1][3]

## 개발자 시점의 워크플로우
- Pre-train: 웹/멀티모달 대규모 데이터로 자기지도 학습해 범용 표현을 획득합니다.[2]
- Adapt: LoRA/PEFT·미세조정·프롬프트 엔지니어링·RAG로 도메인 성능을 끌어올립니다.[3]
- Deploy: GPU 다중 병렬, 클라우드 서비스, 엔터프라이즈 스택에서 추론·모니터링·거버넌스를 운영합니다.[1]

## 언제 어떤 전략을 쓰나
- 도메인 데이터가 적다: 프롬프트 설계 + RAG로 지식 편입을 시도합니다.[3]
- 빠른 PoC: 소형 공개 모델에 LoRA 미세조정으로 비용 최소화합니다.[3]
- 규제/품질 중요: 데이터 정제, 안전·편향 모니터링, 책임있는 오픈 전략을 채택합니다.[4]

## 실제 코드: 3가지 실전 패턴
아래 예시는 로컬 GPU 1~2장 환경을 가정한 학습/추론 빠른 스타터입니다. “작게 시작→점진 확대” 전략을 권장합니다.[3]

### 1) 경량 미세조정(LoRA/PEFT)로 분류 태스크 적응

LoRA(Low-Rank Adaptation)은 대형 기계학습 모델을 특정 용도에 빠르고 효율적으로 적응시키는 기법으로, 원래 모델의 모든 파라미터를 바꾸지 않고 저차원 행렬을 추가하여 미세조정하는 방법입니다. 모델 전체를 재학습하는 대신 소량의 파라미터만 학습해 자원을 크게 절약하면서도 성능 향상을 도모합니다.

PEFT(Parameter-Efficient Fine-Tuning)은 LoRA와 같은 기법을 포함하는, 대형 언어 모델 같은 복잡한 모델을 파라미터를 최소한으로 조정하여 빠르고 적은 자원으로 미세조정하는 전략입니다. 예를 들어, LoRA는 어텐션 레이어의 큰 행렬을 두 개의 저차원 행렬로 분해해 조정 대상 파라미터 수를 줄입니다.

요약하면, LoRA는 PEFT 계열의 기술 중 하나로, 대형모델의 미세조정을 더 빠르고 비용 효율적으로 수행할 수 있게 해 모델을 특정 과제에 특화합니다.

사전학습 LLM을 뉴스 감성 분류에 맞게 LoRA로 미세조정합니다. 파라미터 효율이 높고 VRAM 요구가 낮습니다.[3]
```python
# !pip install transformers datasets peft accelerate bitsandbytes
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

base_model = "mistralai/Mistral-7B-v0.1"  # 예시: 공개 LLM
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model, load_in_4bit=True, device_map="auto"
)

# PEFT/LoRA 구성: 랭크/알파/드롭아웃은 VRAM과 성능 사이 절충
peft_cfg = LoraConfig(
    task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_cfg)

# 예시 데이터: 감성 분류용 텍스트 -> 프롬프트 기반 라벨 생성 태스크로 변환
ds = load_dataset("imdb")
def prompt_map(ex):
    text = ex["text"][:1500]
    label = "positive" if ex["label"] == 1 else "negative"
    ex["input_ids"] = tokenizer(
        f"Review:\n{text}\n\nQuestion: sentiment?\nAnswer:",
        return_tensors=None, truncation=True, max_length=2048
    )["input_ids"]
    ex["labels"] = tokenizer(label, truncation=True, max_length=5)["input_ids"]
    return ex

ds = ds.map(prompt_map, remove_columns=ds["train"].column_names)
args = TrainingArguments(
    output_dir="./out-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_total_limit=2
)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"].select(range(2000)))
trainer.train()

# 추론
inp = tokenizer("Review:\nAwesome movie!\n\nQuestion: sentiment?\nAnswer:", return_tensors="pt").to(model.device)
gen = model.generate(**inp, max_new_tokens=3)
print(tokenizer.decode(gen, skip_special_tokens=True))
```
이 방식은 전체 파라미터가 아닌 어댑터만 학습해, 학습/배포 비용을 크게 낮춥니다. 산업 현장에서는 언어·코드·도메인 지식까지 유사한 패턴으로 확장합니다.[3]

### 2) RAG(Retrieval-Augmented Generation)로 최신성·정확도 강화
다음 코드는 소형 RAG(Retrieval-Augmented Generation) 파이프라인의 최소 예시입니다. 

Retrieval-Augmented Generation (RAG)은 대형 언어 모델(LLM)이 미리 학습된 데이터뿐 아니라 외부 문서나 데이터베이스에서 실시간으로 관련 정보를 검색하고 이를 답변 생성에 반영하는 기술입니다. 이를 통해 최신 정보나 도메인 특정 지식까지 반영할 수 있어 응답의 정확성과 신뢰도가 높아집니다.

구체적으로 RAG는 다음 핵심 절차로 작동합니다:

- **문서 임베딩(벡터 변환)**으로 텍스트를 수치화
- **검색기(Retriever)**를 통해 질문과 관련된 문서 벡터를 찾아내고
- **(선택적) 재정렬기(Reranker)**가 검색 결과의 관련도를 평가

최종적으로 LLM이 검색한 자료를 참고하여 답변을 생성합니다.
이 방법은 AI가 잘못된 내용을 만들어내는 ‘환각(hallucination)’을 줄이며, 모델 재학습 없이 최신 데이터를 반영할 수 있어 비용과 시간을 절감합니다. 또한, 출처를 명시해 투명성과 검증 가능성을 높입니다. 벡터 데이터베이스나 지식 그래프 등의 기술과 결합해 활용되기도 합니다.

흐름은 “문서 임베딩 생성 → 벡터 인덱스 구축 → 질의 임베딩으로 유사 문서 검색 → 검색 컨텍스트를 포함한 프롬프트로 LLM 생성”입니다

파운데이션 모델 위에 벡터 검색을 얹어 최신 사내 문서를 검색·요약·답변합니다. 라벨이 부족한 초기 단계에 특히 유용합니다.[3]
```python
# !pip install faiss-cpu sentence-transformers transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss, numpy as np

docs = [
  ("policy-1", "Security policy v3 updated on July 2025: ..."),
  ("policy-2", "Data retention policy: ..."),
]
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_vecs = embedder.encode([d[1] for d in docs], normalize_embeddings=True)
index = faiss.IndexFlatIP(doc_vecs.shape[1])
index.add(np.array(doc_vecs, dtype=np.float32))

def retrieve(q, k=2):
    qv = embedder.encode([q], normalize_embeddings=True).astype(np.float32)
    D, I = index.search(qv, k)
    return [docs[i] for i in I]

base = "mistralai/Mistral-7B-v0.1"
tok = AutoTokenizer.from_pretrained(base)
llm = AutoModelForCausalLM.from_pretrained(base, device_map="auto")

q = "보안 정책 최신 변경사항 요약해줘."
ctx = "\n\n".join([f"[{i}] {c}" for i, c in retrieve(q)])
prompt = f"다음 컨텍스트만 사용해 답하세요.\n{ctx}\n\n질문: {q}\n답변:"
inp = tok(prompt, return_tensors="pt").to(llm.device)
out = llm.generate(**inp, max_new_tokens=256)
print(tok.decode(out, skip_special_tokens=True))
```

```
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss, numpy as np

문장 임베딩 모델 로더, CausalLM/토크나이저 로더, FAISS와 NumPy를 임포트합니다.

이유: 임베딩(검색용), 벡터 인덱싱(FAISS), 생성 모델(LLM) 호출이 각각 필요하기 때문입니다.
```

```
docs = [
("policy-1", "Security policy v3 updated on July 2025: ..."),
("policy-2", "Data retention policy: ..."),
]

의미: (문서ID, 본문) 형식의 간단한 문서 리스트입니다.

이유: 실제 엔터프라이즈 환경에서는 사내 문서(정책, 가이드 등)를 검색 대상으로 삼습니다. 예시는 최소 데이터로 동작을 보여줍니다.
```

```
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

의미: 경량 문장 임베딩 모델(all-MiniLM-L6-v2)을 불러옵니다.

이유: 빠르고 품질 대비 효율이 좋아 데모·프로토타입에 적합합니다. 검색 품질은 임베딩 모델 선택에 크게 좌우됩니다.
```

```
doc_vecs = embedder.encode([d for d in docs], normalize_embeddings=True)

의미: 각 문서의 본문만 추출해 임베딩 벡터로 변환하고, 정규화(L2 정규화)합니다.

이유: 코사인 유사도 기반 검색을 Inner Product로 바꾸기 위해 보통 임베딩을 정규화하고 내적을 사용합니다. 정규화는 거리/유사도 일관성에 도움을 줍니다.
```

```
index = faiss.IndexFlatIP(doc_vecs.shape)

의미: 내적(Inner Product) 기반의 납작(flat) 인덱스를 생성합니다. 차원 수는 임베딩의 차원을 사용합니다.

이유: 코사인 유사도 ≒ 정규화된 벡터의 내적이므로, 빠르게 유사 문서를 찾기 위해 IP 인덱스를 선택합니다. 초기 데모에서는 간단한 IndexFlatIP가 충분합니다.
```

```
index.add(np.array(doc_vecs, dtype=np.float32))

의미: 문서 임베딩을 float32 배열로 변환해 인덱스에 추가합니다.

이유: FAISS는 float32 입력을 기대합니다. 인덱스에 추가해야 검색이 가능합니다.
```

```
def retrieve(q, k=2):
qv = embedder.encode([q], normalize_embeddings=True).astype(np.float32)
D, I = index.search(qv, k)
return [docs[i] for i in I]

의미: 질의 q를 임베딩해 정규화한 후, 상위 k개 유사 문서를 검색합니다. FAISS는 (거리/유사도, 인덱스) 튜플을 반환하며, 여기선 인덱스로 원본 docs를 되돌립니다.

이유: RAG의 “R(Retrieval)” 단계 핵심입니다. LLM이 컨텍스트로 삼을 근거 문서를 찾습니다. 주의: I는 2차원(배치) 배열이므로 보통 I를 쓰는 것이 안전합니다. 현재 코드는 배치 1 가정에서 작동하지만, 보다 안전하게는 return [docs[i] for i in I]로 수정하는 편이 좋습니다.
```

```
base = "mistralai/Mistral-7B-v0.1"
tok = AutoTokenizer.from_pretrained(base)
llm = AutoModelForCausalLM.from_pretrained(base, device_map="auto")

의미: Mistral-7B 기반의 CausalLM과 토크나이저를 허깅페이스에서 로드합니다. device_map="auto"로 가용 GPU/CPU에 자동 배치합니다.

이유: RAG의 “G(Generation)” 단계에서 검색 컨텍스트를 포함해 답변을 생성합니다. 경량/오픈 모델을 쓰면 로컬 테스트가 용이합니다.
```

```
사용자 질문
q = "보안 정책 최신 변경사항 요약해줘."

의미: 한국어 질의입니다.

이유: 다국어 임베딩/LLM은 한국어도 처리 가능합니다. 임베딩 모델/LLM의 다국어 성능은 선택 모델에 따라 달라집니다.

컨텍스트 구성
ctx = "\n\n".join([f"[{i}] {c}" for i, c in retrieve(q)])

의미: retrieve로 얻은 (id, content) 튜플들을 “[id] 본문” 형식으로 합칩니다.

이유: LLM에 “근거 컨텍스트”를 명시적으로 제공해, 환각을 줄이고 근거 기반 답변을 유도합니다. 프롬프트 내 컨텍스트 표시는 출처 인용을 쉽게 만듭니다.

프롬프트 생성
prompt = f"다음 컨텍스트만 사용해 답하세요.\n{ctx}\n\n질문: {q}\n답변:"

의미: “컨텍스트만 사용”이라는 지시로 외부 지식 사용을 억제하고, 컨텍스트 준수형 답변을 유도합니다. 질문과 답변 섹션을 명확히 구분합니다.

이유: RAG 품질은 프롬프트 지시의 명확성에 크게 좌우됩니다. “컨텍스트 제한”은 환각 억제에 효과적입니다.

토크나이즈 및 디바이스 이동
inp = tok(prompt, return_tensors="pt").to(llm.device)

의미: 프롬프트를 토큰 텐서로 만들고, 모델이 위치한 디바이스(GPU/CPU)로 이동합니다.

이유: 모델과 입력 텐서의 디바이스 일치가 필수입니다. 불일치 시 런타임 에러가 발생합니다.

생성
out = llm.generate(**inp, max_new_tokens=256)

의미: 최대 256 토큰까지 답변을 생성합니다.

이유: 길이 상한을 두어 과도한 생성/비용을 방지합니다. 필요 시 온도, 톱-p, 반복 페널티 등 디코딩 파라미터를 추가해 제어할 수 있습니다.

디코딩
print(tok.decode(out, skip_special_tokens=True))

의미: 생성된 토큰을 문자열로 변환해 출력합니다.

이유: 사람이 읽을 수 있는 최종 답변을 얻습니다. 일반적으로 out은 배치 차원을 갖기 때문에 out를 디코딩하는 것이 관례입니다. 여기서는 프레임워크가 배치 1을 처리해 동작하지만, 명시적으로 out을 쓰는 것이 안전합니다.
```

### 설계 의도 : 

임베딩 + 벡터 검색: LLM 파라미터를 수정하지 않고, 외부 지식을 “검색→프롬프트 주입” 방식으로 반영합니다. 학습 없이 최신성·도메인 적합도를 얻기 위한 가장 경제적인 패턴입니다.

정규화 + Inner Product: 코사인 유사도를 효율적으로 근사하기 위한 전형적 조합입니다. 구현이 단순하고 성능도 안정적입니다.

“컨텍스트만 사용” 프롬프트: 모델의 자유 발상을 억제하고, 제공된 근거 범위 내에서 답하게 하여 환각을 줄입니다.

경량 모델 선택: 로컬·개발 환경에서 빠르게 실험 가능한 모델을 사용해, 반복 속도를 높입니다.

### 실무 팁과 안전한 개선점 :
retrieve 반환 인덱싱: return [docs[i] for i in I] → return [docs[i] for i in I]로 수정하여 배치 차원 안전성 확보.

토큰 길이 관리: 컨텍스트가 길어지면 모델 컨텍스트 윈도우를 초과할 수 있습니다. 문서 chunking과 top-k 조절, 재순위(예: Cross-Encoder) 도입을 고려하세요.

디코딩 제어: temperature=0.2~0.7, top_p=0.9, repetition_penalty 등을 generate에 추가해 일관성과 사실성의 균형을 맞추세요.

평가/로그: 검색된 근거와 함께 답변을 저장해 품질 점검과 회귀 테스트가 가능하도록 하세요.

보안/프라이버시: 사내 문서 사용 시 접근 제어, 로깅 마스킹, PII 필터링을 적용하세요.

RAG는 모델 파라미터를 바꾸지 않고도 최신 정보·사내 지식 반영을 가능하게 하며, 안전성/감사 요구에도 유리합니다.[3]

### 3) 멀티모달(이미지+텍스트) 분류/설명 파이프라인
Foundation Model은 언어를 넘어 비전·멀티모달로 확장됩니다. 이미지 임베딩 + 텍스트 LLM 조합으로 설명/분류를 구현할 수 있습니다.[1]
```python
# !pip install transformers timm
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch

# 1) 비전 임베딩(예: CLIP 계열) -> 2) 임베딩을 텍스트 조건으로 투입하는 간단 파이프라인
img_model_id = "openai/clip-vit-base-patch32"
vis = AutoModel.from_pretrained(img_model_id)
proc = AutoProcessor.from_pretrained(img_model_id)

llm_id = "mistralai/Mistral-7B-v0.1"
tok = AutoTokenizer.from_pretrained(llm_id)
llm = AutoModelForCausalLM.from_pretrained(llm_id, device_map="auto")

img = Image.open("sample.jpg").convert("RGB")
inputs = proc(images=img, return_tensors="pt")
with torch.no_grad():
    vis_emb = vis.get_image_features(**inputs)
desc_prompt = f"이미지 특징 벡터 요약: {vis_emb[:, :8].tolist()} ... 이를 바탕으로 사진을 한 문장으로 설명하세요:"
inp = tok(desc_prompt, return_tensors="pt").to(llm.device)
out = llm.generate(**inp, max_new_tokens=80)
print(tok.decode(out, skip_special_tokens=True))
```
실무에서는 전용 멀티모달 FM(예: 이미지·텍스트 결합 학습)이나 엔드투엔드 프레임워크를 사용해 성능과 안정성을 높입니다.[1]

## 시스템·인프라 고려사항
파운데이션 모델은 GPU 다중 병렬, 고속 인터커넥트, 효율적 메모리 전략이 중요합니다. 추론은 양산 단계 비용의 핵심이므로, 양자화·KV 캐시·서빙 그래프 최적화를 활용합니다.[1]
엔터프라이즈에서는 클라우드·온프레미스·워크스테이션 혼합 배치와 모델 허브/엔드포인트 연계를 고려해 운영 민첩성을 확보합니다.[5]

## 책임·거버넌스 체크리스트
- 데이터: 라이선스·프라이버시·편향 정제, 출처 추적, 도메인 균형 확보.[3]
- 모델: 프롬프트/출력 필터링, 위험 사용 케이스 차단, 지속적 재보정·평가 파이프라인.[1]
- 오픈 전략: 책무성 있는 공개·문서화·리스크 관리 프레임 설계.[4]

## 더 배우기 위한 추천 흐름
- 개념: 스탠퍼드 HAI/CRFM의 FM 개요·정의로 큰 그림을 잡습니다.[2]
- 적용: IBM의 엔터프라이즈 경험과 적은 라벨 시나리오 사례를 참고합니다.[3]
- 역사·응용: NVIDIA의 트렌드/사례·멀티모달 확장 인사이트를 살펴봅니다.[1]

## 마무리
오늘날 FM 실전 전략은 “사전학습 자산을 최대 활용하고, 도메인 적응을 가볍게, 거버넌스는 강하게”입니다. 작은 LoRA부터 RAG, 멀티모달 조합까지 단계적으로 확장해 보시길 권장합니다.[1][3]

## 참고 자료
- Stanford HAI/CRFM: What is a Foundation Model? (정의·개요·임팩트)[2]
- IBM Research: What are foundation models? (엔터프라이즈 적용·라벨 효율)[3]
- NVIDIA Blog(KR): 파운데이션 모델이란/역사·산업 사례·안전성 논의 (멀티모달·운영)[1]
- Responsible Open FM 논의: 오픈·책임·거버넌스 방향성[4]
- NVIDIA AI 파운데이션 모델/엔드포인트·운영 스택 개요[5]

[1](https://blogs.nvidia.co.kr/blog/what-are-foundation-models/)
[2](https://hai.stanford.edu/news/what-foundation-model-explainer-non-experts)
[3](https://research.ibm.com/blog/what-are-foundation-models)
[4](https://hai.stanford.edu/news/how-promote-responsible-open-foundation-models)
[5](https://www.nvidia.com/ko-kr/ai-data-science/foundation-models/)
[6](https://en.wikipedia.org/wiki/Foundation_model)
[7](https://yumdata.tistory.com/400)
[8](https://hai.stanford.edu/news/what-foundation-model-explainer-non-experts?sf177996596=1)
[9](https://www.adalovelaceinstitute.org/resource/foundation-models-explainer/)
[10](https://www.lakera.ai/blog/foundation-models-explained)
[11](https://glossary.zerogap.ai/foundation-model)
[12](https://heidloff.net/article/foundation-models-at-ibm/)
[13](https://encord.com/blog/foundation-models/)
[14](https://developer.nvidia.com/ko-kr/blog/nvidia-ai-foundation-models-build-custom-enterprise-chatbots-and-co-pilots-with-production-ready-llms/)
[15](https://www.techtarget.com/whatis/feature/Foundation-models-explained-Everything-you-need-to-know)
[16](https://www.ibm.com/think/insights/generative-ai-benefits)
[17](https://heidloff.net/article/foundation-models/)
[18](https://blogs.nvidia.co.kr/blog/foundation-models-gaming/)
[19](https://www.linkedin.com/posts/stanfordhai_what-is-a-foundation-model-an-explainer-activity-7071895109943889920-pMVU)
[20](https://lighthouse3.com/our-blog/did-you-know-ibm-has-foundation-models-for-ai/)
[21](https://developer.nvidia.com/ko-kr/blog/enhance-robot-learning-with-synthetic-trajectory-data-generated-by-world-foundation-models/)
[22](https://www.redhat.com/en/blog/ibms-granite-foundation-model-detailed-look-its-training-data)

# Reference
https://yumdata.tistory.com/400
