# Knowledge Distillation 가이드

결론부터 말하면, **Knowledge Distillation(KD)**은 큰 모델(Teacher)의 일반화 능력을 작은 모델(Student)로 전수해 **경량·고속 추론**을 가능하게 하는 모델 압축 기법입니다.[1][2][3]

## 왜 필요한가
대형 모델은 정확하지만 메모리·지연시간 제약이 큰 환경(모바일·엣지·실시간 시스템)에서 배포가 어렵습니다. KD는 큰 모델의 지식을 작은 모델이 모방하도록 학습시켜, 성능 저하를 최소화하면서 추론 효율을 대폭 개선합니다.[2][4][1]

## 역사와 개념
Hinton 등(2015)의 “Distilling the Knowledge in a Neural Network”가 KD를 정식 틀로 제시했습니다. 아이디어는 소프트맥스 온도 $$T$$로 부드러워진 교사 확률분포를 학생이 모방하도록 만들어 “dark knowledge”를 전수한다는 것입니다. 이 접근은 2006년의 모델 압축 계보를 잇는 현대적 프레임으로 자리잡았고, 이후 CV·NLP·음성 등에서 폭넓게 확산되었습니다.[5][4][3][1][2]

## 핵심 수식 요약
- 온도 스케일링 소프트맥스: $$p_i^{(T)} = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)} $$ 로짓 $$z$$, 온도 $$T>1$$일수록 분포가 더 “soft”해집니다.[3][1]
- KD 손실 결합: 학생의 하드 라벨 분류 손실과 교사-학생 간 소프트 분포 모방 손실을 가중합합니다. 전형적으로 $$\mathcal{L} = \alpha \cdot \mathcal{L}\_{CE}(y, s) + (1-\alpha)\cdot T^2 \cdot \mathcal{L}\_{KD}(p_t^{(T)}, p_s^{(T)})$$ 형태가 널리 쓰입니다. 여기서 $$\mathcal{L}_{KD}$$로 KLDiv 또는 CE를 사용하고, $$T^2$$ 보정이 일반적입니다.[6][3]

## 온도 T와 가중치 α 선택
- $$T$$: 너무 낮으면 분포가 딱딱해져 부가 신호가 사라지고, 너무 높으면 구분력이 약해질 수 있습니다. 중간 정도의 $$T$$가 전체 성능을 개선하는 경향이 여러 실험에서 관찰됩니다. 실무에선 2~5 범위를 자주 시작점으로 둡니다.[7][6][3]
- $$\alpha$$: 데이터 라벨 신뢰가 높으면 학생 CE 비중(α)을 키우고, 교사 모방 효과를 키우려면 1-α 쪽을 늘립니다. 보편적 초기값으로 α≈0.1~0.5가 자주 쓰입니다.[6][3]

## 구현 패턴
- 교사(고정)·학생(학습) 구조, 추가 손실만 얹으면 기존 파이프라인에 쉽게 통합할 수 있습니다.[3][6]
- 프레임워크 예제: Keras Distiller 클래스로 α, T를 지정하고 teacher/student 로짓으로 distillation_loss를 구성합니다. PyTorch 튜토리얼도 학생 CE + KD 손실 결합을 단계별로 보여줍니다.[6][3]

## PyTorch 예시 코드(MNIST/CIFAR로 확장 가능)
다음 코드는 교사/학생을 정의하고, KD 손실을 적용해 학습하는 최소 구현입니다. 연구 코드에 바로 이식할 수 있도록 구성했습니다.[8][3]

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim

# 1) 모델 정의: 간단한 교사/학생 (실전에서는 교사=ResNet18/50, 학생=작은 CNN 권장)
class StudentNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.features(x)       # [B, 64, 1, 1]
        x = x.flatten(1)           # [B, 64]
        return self.fc(x)          # logits

class TeacherNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 예: 사전학습 가중치 대체 가능. 입력 채널 맞춤 필요 시 첫 Conv 수정
        self.backbone = models.resnet18(num_classes=num_classes)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, x):
        return self.backbone(x)    # logits

# 2) KD 손실 함수: T^2 보정 + α 가중 결합
def kd_loss_fn(student_logits, teacher_logits, hard_labels, T=4.0, alpha=0.2):
    # 학생 하드 라벨 CE
    ce = F.cross_entropy(student_logits, hard_labels)
    # 소프트 분포 KL (log_softmax vs softmax)
    p_s_T = F.log_softmax(student_logits / T, dim=1)
    p_t_T = F.softmax(teacher_logits / T, dim=1)
    kd = F.kl_div(p_s_T, p_t_T, reduction="batchmean")
    # T^2 스케일 보정 및 가중합
    return alpha * ce + (1 - alpha) * (T * T) * kd

# 3) 데이터 로더 (MNIST)
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4) 모델 생성
teacher = TeacherNet(num_classes=10).to(device)
student = StudentNet(num_classes=10).to(device)

# 5) 교사 고정(사전학습된 가중치 로드 권장)
# 예: teacher.load_state_dict(torch.load("teacher.pt"))
for p in teacher.parameters():
    p.requires_grad = False
teacher.eval()

# 6) 최적화 설정
optimizer = optim.Adam(student.parameters(), lr=1e-3)

# 7) 학습 루프
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

epochs = 5
T, alpha = 4.0, 0.2

for epoch in range(1, epochs + 1):
    student.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            t_logits = teacher(x)
        s_logits = student(x)
        loss = kd_loss_fn(s_logits, t_logits, y, T=T, alpha=alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = evaluate(student, test_loader)
    print(f"Epoch {epoch} | KD Acc: {acc:.4f}")
```
이 구현은 PyTorch 공식 튜토리얼의 원리를 따르며, KLDiv를 기반으로 $$T^2$$ 스케일을 포함해 α 가중 결합을 적용합니다. CIFAR-10 등으로 확장하려면 입력 채널과 변환(정규화, 데이터 증강) 및 백본을 조정하면 됩니다.[9][8][3]

## Keras 스타일 구현 개요
Keras 예제는 Distiller 래퍼 모델을 정의해, compile 단계에서 α와 T를 주고 compute_loss에서 teacher/student 예측을 소프트맥스로 변환해 distillation_loss를 계산합니다. 최종 손실은 α·student_loss + (1-α)·distillation_loss이며, distillation 부분에 $$T^2$$를 곱합니다. 텐서플로/케라스 환경일 경우 이 패턴이 가장 빠른 시작점입니다.[6]

## 실전 팁
- 교사 선택: 과대적합된 교사보다 일반화가 좋은 교사가 더 효과적입니다. 앙상블 교사를 단일 학생으로 증류하는 것도 강력합니다.[1][2]
- 로짓 사용: 라벨 대신 로짓 기반 소프트 타깃을 쓰면 클래스 간 유사도를 반영한 풍부한 신호를 제공합니다.[1][3]
- 하이퍼스윕: $$T\in\{2,3,4,5\}$$, $$\alpha\in\{0.1,0.3,0.5\}$$ 정도 그리드 탐색을 권장합니다. 적정 $$T$$가 성능·안정성·강건성의 균형을 높일 수 있습니다.[7][3]
- 전이/파인튜닝 병행: 도메인 시프트가 있으면 학생에 라벨 기반 파인튜닝을 병행해 KD의 모방 오차를 상쇄합니다.[2][3]

## 변형·응용
- 로지트/특징 증류: 로지트 모방 외에 중간 특징지도/어텐션 맵 정렬로 성능을 높입니다(“deep” KD 계열 구현 예시 다수).[8][2]
- NLP: DistilBERT처럼 사전학습된 대형 언어모델을 경량화하는 대표 사례가 다수 보고되었습니다.[10][2]
- LLM 파인튜닝: torchtune 튜토리얼은 Llama 계열 간 증류 워크플로를 제공합니다(응용 확장 참고).[11]

```
로지트(logit)란 머신러닝, 특히 분류 문제에서 모델이 출력하는 확률값 이전의 원시 점수를 말합니다.
이는 소프트맥스 함수에 입력되어 각 클래스의 확률로 변환됩니다.
지식 증류(knowledge distillation)에서는 교사 모델의 로지트 출력을 학생 모델에 전달해 학습시키며, 이렇게 하면 단순한 정답 레이블보다 더 풍부한 정보를 전달할 수 있습니다.

특징 증류는 모델의 중간층 특징 맵이나 내부 표현을 학생 모델에게 전달해 학습하는 방법입니다.
학생 모델은 교사 모델의 중간 특성들을 보면서 더 깊고 효과적인 학습이 가능해집니다.
예를 들어, 여러 깊이의 중간 레이어에 어텐션 기반 얕은 분류자를 추가해 학습하는 방법이 있습니다.
```

## 참고 자료
- Hinton et al., Distilling the Knowledge in a Neural Network(2015): KD의 정식 틀과 dark knowledge 개념 제시.[5][1]
- PyTorch 튜토리얼: 학생 CE + KD 결합 손실의 표준 구현과 실험 전략.[9][3]
- Keras 예제: Distiller 클래스로 α·T 제어하는 실전 템플릿.[6]
- 개념·응용 개요: 네프튠 블로그, IBM 개요, 역사·사례 정리.[4][2]

이 글을 바탕으로, 소형 모델의 정확도와 효율을 동시에 끌어올리는 KD를 실험 설계에 바로 적용할 수 있습니다. 핵심은 적합한 **T**와 **α**를 잡고, 교사의 유용한 분포 정보를 학생이 충분히 흡수하도록 학습 루프를 안정적으로 구성하는 것입니다.[3][6]

[1](https://arxiv.org/abs/1503.02531)
[2](https://neptune.ai/blog/knowledge-distillation)
[3](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
[4](https://www.ibm.com/think/topics/knowledge-distillation)
[5](https://arxiv.org/pdf/1503.02531.pdf)
[6](https://keras.io/examples/vision/knowledge_distillation/)
[7](https://arxiv.org/html/2502.20604v1)
[8](https://github.com/haitongli/knowledge-distillation-pytorch)
[9](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
[10](https://zerojsh00.github.io/posts/DistilBERT/)
[11](https://docs.pytorch.org/torchtune/0.3/tutorials/llama_kd_tutorial.html)
[12](https://baeseongsu.github.io/posts/knowledge-distillation/)
[13](https://www.semanticscholar.org/paper/Distilling-the-Knowledge-in-a-Neural-Network-Hinton-Vinyals/0c908739fbff75f03469d13d4a1a07de3414ee19)
[14](https://seungwooham.tistory.com/entry/Distilling-the-Knowledge-in-a-Neural-Network-%EC%9A%94%EC%95%BD-%EB%B0%8F-%EC%84%A4%EB%AA%85)
[15](https://deep-learning-study.tistory.com/700)
[16](https://jamiekang.github.io/2017/05/21/distilling-the-knowledge-in-a-neural-network/)
[17](https://scholar.google.com/citations?user=JicYPdAAAAAJ&hl=ko)
[18](https://littlefoxdiary.tistory.com/134)
[19](https://tutorials.pytorch.kr/beginner/knowledge_distillation_tutorial.html)
[20](https://www.kaggle.com/code/shivangitomar/knowledge-distillation-part-1-pytorch)
[21](https://labelyourdata.com/articles/machine-learning/knowledge-distillation)

# Reference
https://baeseongsu.github.io/posts/knowledge-distillation/
