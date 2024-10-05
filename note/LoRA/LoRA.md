# LoRA ⭐

---

<aside>
💡 **LoRA** Low-Rank Adaptation

---

GPT-3와 같은 대형 언어 모델을 특정 작업에 맞게 효율적으로 적응시키기 위한 새로운 방법론인 LoRA 

- 사전 학습된 가중치 행렬의 업데이트를 저랭크 행렬로 표현
</aside>

- 용어
    
    👐🏻 **단어 임베딩** Word Embedding
    
    - 문자를 숫자들의 배열인 벡터로 변환하는 방법으로 만들어진 단어 벡터 의미
        
        
        |  | 원-핫 벡터 | 임베딩 벡터 |
        | --- | --- | --- |
        | 차원 | 고차원(단어 집합의 크기) | 저차원 |
        | 다른 표현 | 희소 벡터의 일종 | 밀집 벡터의 일종 |
        | 표현 방법 | 수동 | 훈련 데이터로부터 학습함 |
        | 값의 타입 | 1과 0 | 실수 |
    
    👐🏻 **희소 표현** sparse representation
    
    - 벡터 또는 행렬의 값이 대부분이 0으로 표현되는 방법
    
    👐🏻 **분산 표현** distributed representation
    
    - 단어의 의미를 다차원 공간에 벡터화하는 방법을 사용
    - 분산 표현을 이용하여 단어 간 의미적 유사성을 벡터화하는 작업을 워드 임베딩(embedding)

---

## 0. Abstract

---

> **LoRA: Low-Rank Adaptation of Large Language Models**
> 
> - LLM에서 튜닝할 때, 모델의 특정 가중치(weight) 업데이트를 저랭크 행렬 분해 방식으로 표현

자연어 처리에서 중요한 패러다임은 아래 두가지로 구성

- large-scale pretraining on general domain data
    
    일반 도메인 데이터에 대한 대규모 사전학습  
    
- adaptation to particular tasks or domains
    
    특정 task나 도메인에 대한 적응
    

이때, 더 큰 모델을 사전 학습할수록 모든 모델 파라미터를 재학습하는 완전한 파인튜닝(fine-tuning)이 힘들어짐따라서 Low-Rank Adaptation(LoRA)을 제안

1️⃣ freeze the pretrained model weights 

- "사과는 달콤하다"와 같은 문장의 의미를 이해
- 감정 분석과 관련이 없더라도 여전히 이러한 기본 언어 이해 능력 보유

2️⃣ provide an empirical investigation into rank-deficiency in language model adaptation

트랜스포머 아키텍처의 각 층에 학습 가능한 저랭크 행렬을 주입해 특정 태스크에 관련된 패턴을 학습

- "서비스가 끔찍했다"는 문장을 만났을 때,
    
    행렬은 '끔찍하다'라는 단어가 부정적인 감정을 나타낸다는 것을 학습
    

LoRA는 RoBERTa, DeBERTa, GPT-2, GPT-3 모델의 품질에서 파인튜닝과 동등하거나 더 나은 성능을 보임

더 적은 학습 가능한 파라미터를 가지면서도 더 높은 학습 처리량을 제공

## 1. Introduction

---

### Problem Statement

1️⃣ **Too many parameter variables**

![[https://blog.dailydoseofds.com/p/full-model-fine-tuning-vs-lora-vs](https://blog.dailydoseofds.com/p/full-model-fine-tuning-vs-lora-vs)](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/52844fcc-6094-4fdb-ba8c-52737ab9c821_1640x402.gif)

[https://blog.dailydoseofds.com/p/full-model-fine-tuning-vs-lora-vs](https://blog.dailydoseofds.com/p/full-model-fine-tuning-vs-lora-vs)

- 파인튜닝은 사전 학습된 모델의 ‘모든’ 파라미터를 업데이트하는 방식
- 모델이 클수록 배포 시 엄청난 저장 공간과 계산 비용이 필요

2️⃣ 추론 지연 inference latency

![**GPT-2 medium 모델의 추론 시 어댑터 레이어가 추가되는 경우 지연 시간이 증가**
LoRA는 어댑터 레이어와 비교하여 지연 시간이 적게 증가](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-24_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_11.20.31.png)

**GPT-2 medium 모델의 추론 시 어댑터 레이어가 추가되는 경우 지연 시간이 증가**
LoRA는 어댑터 레이어와 비교하여 지연 시간이 적게 증가

- 1️⃣ 보완을 위해 모든 파라미터를 적응시키지 않고 일부 파라미터만 적응시키거나, 외부 모듈을 추가하여 새로운 작업에 적응시키려는 시도 존재
- 모델의 깊이를 확장하거나 추가적인 모듈을 삽입함으로써 새로운 작업에 적응할 때 추가적인 추론 지연(inference latency)을 유발
    
    * 추론 지연 : 모델이 입력 데이터를 처리하여 출력을 생성하는 데 걸리는 시간
    
- 실시간 성능이 중요한 애플리케이션에서 문제 발생할 수 있음

### Existing Solutions

**1️⃣ Adapter layer 추가**

→  학습 이후 생성의 속도가 느려지는 **Inference Latency** 현상이 발생

생성 시퀀스의 길이가 길어질수록 생성 속도가 유의미하게 차이가 발생함

**2️⃣ Prefix Tuning 최적화**

- **Prefix tuning**은 모델의 입력에 특정 프리픽스(즉, 고정된 길이의 토큰 시퀀스)를 추가하여, 모델이 새로운 작업에 적응하도록 하는 방법
    
    → 성능이 일관되게 증가하거나 감소하지 않고, 예상치 못하게 오르락내리락함
    

### **🔎 Low-Rank Adaptation(LoRA)**

![[https://blog.dailydoseofds.com/p/full-model-fine-tuning-vs-lora-vs](https://blog.dailydoseofds.com/p/full-model-fine-tuning-vs-lora-vs)](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/ebf6d60c-7495-4039-a617-6447c1a06d8e_1640x798.webp)

[https://blog.dailydoseofds.com/p/full-model-fine-tuning-vs-lora-vs](https://blog.dailydoseofds.com/p/full-model-fine-tuning-vs-lora-vs)

모델의 모든 가중치를 학습하는 대신, 모델의 특정 층에 저랭크 행렬을 주입하여 가중치 업데이트를 간접적으로 최적화

- **기존 모델의 가중치 Freeze**
    - 사전 학습된 모델의 가중치(예: GPT-3의 가중치)를 고정(freeze)

---

- **저랭크 행렬 도입**
    - 모델의 각 층에 저랭크(low-rank) 행렬 A와 B를 추가

---

- **저랭크 행렬의 학습**
    - 학습 과정에서 저랭크 행렬 A와 B만을 학습
    - 기존의 큰 가중치 행렬 W0는 고정

![image.png](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/image.png)

📍 **장점**

1. **메모리와 계산 비용 절감**
    - 학습 가능한 파라미터 수를 크게 줄이기 때문에, 더 적은 메모리로 모델을 훈련
2. **빠른 작업 전환**
    - LoRA를 사용하면 작업 간 전환이 매우 빠름
    - 새로운 작업에 맞게 A와 B만 교체하면 되기 때문에, 전체 모델을 다시 학습하거나 로드할 필요가 없음
3. **추가적인 추론 지연 없음**
    - LoRA는 학습할 때만 저랭크 행렬을 사용하고, 모델이 배포될 때는 원래 가중치와 저랭크 행렬을 미리 계산하여 합쳐모델이 실제로 추론을 수행할 때 추가적인 계산 지연이 발생하지 않음

## 2. Architecture

---

<aside>
<img src="https://www.notion.so/icons/notification_lightgray.svg" alt="https://www.notion.so/icons/notification_lightgray.svg" width="40px" /> **LOW-RANK-PARAMETRIZED UPDATE MATRICES**

---

- **Dense Layers & Full-Rank  Weights**
    - 신경망에는 여러 밀집 계층(dense layers)이 있으며, 이 계층들은 행렬 곱셈을 수행.
    - 일반적으로, 이 밀집 계층의 가중치 행렬은 풀랭크(full-rank)
    - 풀랭크 가중치란, 행렬이 최대 차원의 정보를 담고 있음
- **Low-Rank Hypothesis**
    - 사전 학습된 언어 모델은 고유 차원(intrinsic dimension) 낮음.
        - 즉, 더 작은 부분 공간으로 투영하더라도 여전히 효율적으로 학습 가능.
    - LoRA는 특정 task에 적응할 때 가중치 업데이트도 "고유 랭크(intrinsic rank)"가 낮을 것이라고 가정
    
    ![[https://www.aporia.com/learn/low-rank-adaptation-lora/](https://www.aporia.com/learn/low-rank-adaptation-lora/)](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/image%201.png)
    
    [https://www.aporia.com/learn/low-rank-adaptation-lora/](https://www.aporia.com/learn/low-rank-adaptation-lora/)
    
- **Low Rank**
    
    $W_0 +ΔW=W_0 +BA$
    
    - $W_0$와 $BA$의 합으로 계산된 $W$를 사용하여 추가적인 계산 없이 추론을 수행해, 지연이 발생하지 않음
    - $W_0$는 사전 학습된 모델의 가중치 행렬로 크기 $d × k$
    - $B$를 $d × r$, $A$를 $r × k$ 의 차원으로 분해해 $BA$를 이용하여 업데이트가 가능
        - 이때 B, A는 원래 가중치 행렬 $W_0$ 에 비해 훨씬 작은 차원을 갖고 있어 학습해야 할 파라미터 수가 크게 줄어듦
- **훈련 과정에서의 고정(Freeze)와 업데이트**
    - 사전 학습된 가중치 행렬 $W_0$는 훈련 과정에서 고정(freeze)되어, 학습 도중 그래디언트 업데이트되지 않음
    - 사전 학습된 모델의 지식을 그대로 유지하면서, 학습 가능한 파라미터인 저랭크 행렬 $A$와 $B$만을 업데이트
    - 모델의 출력 조정식 → 업데이트 수식
        
        $h = W_0x +ΔWx=W_0x +BAx$
        
        - $x$ : 입력 벡터 / $h$ : 출력벡터
- **초기화와 스케일링**
    - $A$  가우시안 분포로 초기화 / $B$ 0으로 초기화
    - 저랭크 행렬의 효과를 조절하기 위해 $ΔWx$ 을 $\frac{\alpha}{r}$로 스케일링
        - $α$는 상수로, 저랭크 행렬의 랭크에 따라 학습율을 조절
            
            Adam으로 최적화할 때 $\alpha$를 튜닝하는 것은 learning rate 를 튜닝하는 것과 거의 동일
            
        - 저행렬 랭크 $r$
            
            스케일링은 r을 변경할 때 파라미터를 다시 조정하지 않아도 되게 함
            
</aside>

<aside>
<img src="https://www.notion.so/icons/notification_gray.svg" alt="https://www.notion.so/icons/notification_gray.svg" width="40px" /> **APPLYING LoRA TO TRANSFORMER**

---

**parameter-efficiency를 위해 downstream task에 대해서 attention weight만 adapting하고 MLP에서는 동결**

![[https://aifactory.space/task/2733/discussion/934](https://aifactory.space/task/2733/discussion/934)](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/image%202.png)

[https://aifactory.space/task/2733/discussion/934](https://aifactory.space/task/2733/discussion/934)

- **Self-Attention 모듈** $W_q, W_k, W_v, W_o$
    - 쿼리, 키, 값, 출력에 해당하는 행렬이 바로 LoRA가 적용되는 주요 부분
    - 해당 행렬의 업데이트를 저랭크 방식으로 표현하여 학습 효율성을 높임
    - self attention 중 선택하여 ( LoRA_B x LoRA_A ) 를 단순히 더함
- 파라미터 수를 최소화하고 학습을 단순화하기 위해 **MLP 모듈의 가중치 행렬은 동결(freeze)**
    - MLP 모듈은 다운스트림 작업에서 학습되지 않음
- **Multi-Head Attention** **모듈**
    - 여기서 가중치 행렬을 Freeze하고, 저랭크 행렬만을 학습함으로써 필요한 계산 자원과 메모리를 줄임
</aside>

## 3.  Experiments

---

> 📓 **실험 정리**
> 
> - **Model** | RoBERTa, DeBERTa, GPT-2, GPT-3
> - **Task |** NLU(자연어 이해), NLG(자연어 생성), GLUE(BERTa)
> - **Baseline**
>     - FineTuing
>     - Bias-only BitFit
>     - Prefix-embedding tuning
>     - Prefix-layer tuning
>     - Adapter tuning
>     - LoRA ⭐

![스크린샷 2024-08-24 오전 11.31.36.png](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-24_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_11.31.36.png)

- Encoder류 성능 비교
- FT과 비견되게 좋음, Adapter 보다 좋음
- FT 파라미터 125M, LoRA 파라미터 0.3M

![스크린샷 2024-08-24 오전 11.31.50.png](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-24_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_11.31.50.png)

- Decoder류 성능 비교
- 성능 좋음

![스크린샷 2024-08-24 오전 11.32.04.png](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-24_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_11.32.04.png)

- 서로 다른 methods 비교
- 학습가능한 파라미터 수가 적음에도 불구하고 성능이 좋음

![스크린샷 2024-08-24 오전 11.32.19.png](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-24_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_11.32.19.png)

- 다른 튜닝방법들의 경우 일정 토큰수 이상이 넘어가게 되면 오히려 성능이 하락하는 결과를 보여줬음에도 LoRA는 그렇지 않고 비교적 일정한 성능을 보이는 것을 확인

**➕ 어떤 가중치에 적용하는 것이 좋을까**

![스크린샷 2024-08-24 오후 9.01.03.png](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-24_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_9.01.03.png)

- 2 > 4 > 8 순서대로 적은 차원에서 성능 좋음
- rank가 클수록 더 많은 표현력을 가지도록 학습할 수 있지만, 더 많은 파라미터를 사용
- 단일 가중치에 높은 rank를 적용한 것 보다 여러 가중치에 낮은 rank를 적용한 것이 더 좋은 성능
- (q,k) 보다는 (q,v)가 성능이 좋고 다 적용할 때 성능이 좋음

![스크린샷 2024-08-24 오후 9.06.47.png](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-24_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_9.06.47.png)

- 매우 작은 rank로도 준수한 성능을 내는 것을 확인

**두 가지 다른 랭크 r=8과 r=64에 대한 서브스페이스(subspace) 유사도를 비교**

![스크린샷 2024-08-24 오후 9.40.49.png](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-24_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_9.40.49.png)

- **쿼리**와 **값** 가중치 행렬이 학습 중에 어떻게 업데이트되었는지 알 수 있음
    
    → 모델의 특정 부분에 어떤 영향을 미치는지를 이해 가능
    

![스크린샷 2024-08-24 오후 9.41.00.png](LoRA%20%E2%AD%90%202eb4923f4efc45e2a5fc7f86333220ee/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-08-24_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_9.41.00.png)

## 4. Conclusion

---

- LLM을 fine-tuing하는 것은 자원이 많이 듦
- 성능 및 효율성 향상
    - LoRA를 사용해 추론 지연을 발생시키지 않고 시퀀스 길이를 유지해 높은 모델 품질 유지 가능
- 모든 Dense layer에 적용 가능
    - Low-Rank 행렬을 사용해 모델 파라미터 업데이트를 수행하는 새로운 전략 제시
- 모델의 파라미터 대부분을 공유함으로써 서비스 배포 시, 신속한 Task 전환 가능

## 🔎 궁금증 ..

---

## 🎹 코드 리뷰

---

[Google Colab](https://colab.research.google.com/drive/1eM5SxPGK3jIxhmUesLmEf-kNJPdH_-Wo?usp=sharing)

## ➕ 참고

---

[LLM 논문리뷰📎 LoRA: Low-Rank Adaptation of Large Language Models](https://www.youtube.com/watch?v=_pshV5XuJjk&t=467s)

[https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)

→ LoRA를 PyTorch 모델에 통합할 수 있는 패키지를 출시하고, RoBERTa, DeBERTa, GPT-2에 대한 구현과 모델 체크포인트 제공