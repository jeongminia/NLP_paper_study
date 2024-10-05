# QLoRA ⭐

---

<aside>
💡 **QLoRA** Quantized Low Rank Adapters
****대형 언어 모델 LLM 을 효율적으로 미세 조정 fine-tuning 하기 위한 접근법이며, LoRA 방식과 함께 양자화를 적용

</aside>

---

## 0. Abstract

---

> **QLoRA: Efficient Finetuning of Quantized LLMs**
> 
> - 양자화된 대형 언어 모델(LLM)의 효율적인 미세 조정

대형 모델은 엄청난 양의 파라미터를 가지고 있음. 

일반적인 미세 조정 방법을 적용하면 매우 큰 GPU 메모리 용량과 시간이 소요. 

**⇒ QLoRA**는 이 문제를 해결하면서도 성능 저하 없이 대형 모델을 미세 조정할 수 있는 방법을 제시.

## 1. Introduction

---

**기존 방법론의 문제**

- Large Model의 파인튜닝 작업이 시간과 비용 소요가 너무 큼
- 일반적인 finetuning을 적용하면 매우 큰 GPU 메모리 용량과 시간이 소요
- 양자화(quantization) 기법들은 추론 단계에서 메모리 사용량을 줄이는 데는 성공했으나 학습 과정에서는 제대로 작동하지 않는 한계

⇒ **LoRA와 Quantization을 적용**한 **QLoRA** 도입

<aside>
🔥

**QLoRA**  Quantized Low Rank Adapters

---

**목표**

- QLoRA는 대형 모델을 4비트 양자화로 finetuning하는 방법 제안
- GPU 메모리 요구량을 획기적으로 감소하며 성능 유지
- ex. 65B 파라미터 모델의 finetuning
    
    평균 메모리 요구량을 780GB 이상에서 48GB 미만으로 줄이면서도 실행 시간이나 성능을 저하 X
    

---

**구현 기법**

QLoRA는 성능을 유지하면서 메모리 사용량을 낮추기 위해 다음과 같은 기술 도입

![스크린샷 2024-09-26 오후 5.39.15.png](QLoRA%20%E2%AD%90%2010d86814722780099048c1b2abfdfcc8/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-26_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_5.39.15.png)

`4-bit NormalFloat Quantization`

- QLoRA는 데이터를 4비트로 양자화하며, NormalFloat라는 데이터 타입을 사용
- 기존의 4비트 정수 또는 부동소수점 방식보다 더 나은 성능을 보임

`Double Quantization`

- 양자화된 상수들도 다시 양자화하는 방식
- 각 파라미터당 더 절약 가능

`Paged Optimizers`

- NVIDIA의 통합 메모리를 사용해 긴 시퀀스를 처리할 때 발생하는 급증현상 Memory Spike를 방지
- 일반적인 방법은 미니 배치를 처리할 때 메모리 요구량이 급격히 증가하
- QLoRA는 이를 효과적으로 방지하여 메모리 사용량을 줄임

`Low-Rank Adapter`

[지난 스터디 복기](https://www.notion.so/LoRA-2eb4923f4efc45e2a5fc7f86333220ee?pvs=21)

- 모델의 각 층마다 어댑터를 추가하여 적은 파라미터로 성능을 유지하는 방식
- 사전 학습된 가중치 행렬의 업데이트를 저랭크 행렬로 표현

---

**기대효과**

- Memory Overhead로 인해 일반적인 미세 조정으로는 불가능한 모델 스케일에서
    
    instruction 기반 finetuning과 chatbot performance에 대한 심층적인 연구를 수행 
    
- 메모리 효율성 향상과 계산 효율성 향상

---

**Experiments 요약**

- QLoRA로 훈련된 Guanaco 모델
    - 최신 챗봇 성능 벤치마크에서 매우 높은 성과를 보임
- 데이터 품질과 데이터셋 적합성이 모델 성능에서 중요한 역할
    - 대형 모델의 성능을 평가할 때에는 인간 평가와 GPT-4와 같은 모델 평가를 병행하는 것이 유용
</aside>

## 2. Model Architecture

---

### Background

**Quantization** | Block-wise k-bit Quantization 

- 양자화는 모델의 파라미터를 더 적은 비트로 표현하여 메모리를 절약하는 방법
- BUT. 이상치를 처리할 때 일부 데이터가 제대로 표현되지 않는 문제가 발생
    - 참고
        
        ![[https://www.youtube.com/watch?v=0jsOaPhoQXs](https://www.youtube.com/watch?v=0jsOaPhoQXs)](QLoRA%20%E2%AD%90%2010d86814722780099048c1b2abfdfcc8/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-27_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_12.29.48.png)
        
        [https://www.youtube.com/watch?v=0jsOaPhoQXs](https://www.youtube.com/watch?v=0jsOaPhoQXs)
        

⏩ 해결방안 . **블록 기반 양자화**(Block-wise Quantization) 방법을 사용 

- 데이터를 여러 개의 블록으로 나눈 후 각 블록을 독립적으로 양자화하는 방식 채택
- 이때, 하나의 배치를 $n=(b×h)/B$개의 블록으로 나누어 각각에 대하여 quantization을 수행

**LoRA** | Low-rank Adapters (LoRA)

![[https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html)](QLoRA%20%E2%AD%90%2010d86814722780099048c1b2abfdfcc8/image.png)

[https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html)

- 대형 모델을 fine-tuing하는 데 사용되는 방법
- 전체 모델 파라미터를 학습하는 대신, 일부 중요한 파라미터(adatper)를 학습해 메모리와 계산 비용 감소
    
    →  역전파 시, 미리 학습된 모델 가중치는 고정된 상태로 유지되며, 어댑터만 업데이트
    
    $Y = XW + s X L_1 L_2$
    
    $L_1 \in \mathbb{R}^{h \times r}, \quad L_2 \in \mathbb{R}^{r \times o}, \quad s \text{는 스칼라 값}$
    
- 장점. 작은 파라미터 세트를 사용하여 모델을 빠르게 미세 조정 가능
- 장점. 대규모 언어 모델에서 메모리 요구량을 크게 줄일 수 있음

**Memory Requirements** | Memory Requirement of Parameter-Efficient Finetuning

- 학습 도중 Backpropagation을 할 때 사용하는 활성화 그래디언트는 매우 많은 메모리 필요
    - 학습 과정에서 각 층의 출력값이 그래디언트를 계산하기 위해 저장되기 때문
    - 활성화 그래디언트는 모델의 크기와 시퀀스 길이에 비례해 큰 메모리를 차지
    - 모델이 추론을 위해 다양한 연산을 수행할 때, 각 연산의 중간결과를 저장해야 함

⏩ 해결방안으로 **그래디언트 체크포인팅 Gradient Checkpointing** 도입

- 활성화 그래디언트가 사용하는 메모리를 줄이기 위한 기술
- 활성화값을 매번 저장하지 않고 필요할 때 다시 계산하는 방식으로 메모리 소모 줄임
- 연산 시간을 약간 증가시키는 대신 메모리 효율 향상

### QLoRA Finetuning

1️⃣ **4-bit NormalFloat Quantization**

- NormalFloat(NF) 데이터 타입은 Quantile Quantization 백분위 양자화 에 기반을 두고 있음
    - But. 백분위 양자화의 한계는 백분위 계산 과정이 매우 비용이 큼 ⇒ 따라서 백분위 근사 알고리즘 적용
- 가중치를 일정 범위로 정규화해 데이터 타입의 범위에 정확히 맞추는 방식으로 처리
- **[-1, 1]** 범위 내로 값을 정규화하여 양자화 진행
- 정규 분포에 대한 정보 이론적으로 최적화된 데이터 타입은 아래와 같이 계산
    
    $$
    q_i = \frac{1}{2} \left( Q_X \left( \frac{i}{2^k + 1} \right) + Q_X \left( \frac{i + 1}{2^k + 1} \right) \right)
    $$
    
    - $Q_x$ : 표준 정규 분포에 대한 백분위 함수

2️⃣ **Double Quantization**

- 추가적인 메모리 절약을 위해 양자화 상수를 양자화하는 방식
- 블록 사이즈가 작을수록 정밀한 양자화를 위해 더 많은 메모리 필요
    
    → 양자화 상수를 한번 더 양자화해 메모리 사용량 감소
    

3️⃣ **Paged Optimizers**

- NVIDIA 통합 메모리를 사용해 CPU와 GPU 간의 페이지 이동이 발생하는 상황에서 메모리 오류 방지하는 기능 제공
- 메모리 상태를 CPU RAM으로 자동으로 대피시켜 GPU 메모리가 부족할 때 메모리 사용량 최적화

⏩ **QLoRA**

- 단일 선형 레이어에서 다음과 같이 **QLORA**를 정의
    
    $$
    Y^{\text{BF16}} = X^{\text{BF16}} \, \text{doubleDequant}(c_2^{\text{FP32}}, \text{k-bit}, W^{\text{NF4}}) + X^{\text{BF16}} L_1^{\text{BF16}} L_2^{\text{BF16}}
    $$
    
- **doubleDequant**는 다음과 같이 정의되며 파라미터 업데이트는 어댑터 가중치에 대해 진행되며, 4비트 가중치에는 영향을 미치지 않음
    
    

## 3.  Experiments

---

### QLoRA vs. Standard Finetuning

일부만 파라미터튜닝하며 양자화를 적용하는 QLoRA가 전체 모델 미세 조정만큼 성능을 발휘할 수 있는가?

- NormalFloat4(NF4)의 영향을 분석
- 아래 실험을 통해 4bit-QLoRA가 다양한 크기, 작업 및 데이터셋에서 16bit 성능을 일치함을 알 수 있음

**Experimental setup**

- Model 3가지 아키텍쳐
    - Encoder ⇒ RoBERTa-large
    - Encoder-Decoder ⇒ T5
    - Decoder ⇒ LLM
    
    → 3B(30억) 파라미터 크기의 모델에서 16bit Adapter finetuning과 QLoRA finetuning을 비교
    
- Benchmark :  GLUE, Super-NaturalInstructions etc.
- ~~Paged optimizers :~~
    
    ~~긴 시퀀스를 처리하는 미니 배치에서만 페이지 처리가 발생하므로 일반적인 상황에서는 이와 관련한 측정값을 제공하지 않음~~
    

🔎 **Default LoRA hyperparameters do not match 16-bit performance**

기본 LoRA 하이퍼파라미터는 16비트 성능에 맞지 않음

![스크린샷 2024-09-27 오후 1.17.00.png](QLoRA%20%E2%AD%90%2010d86814722780099048c1b2abfdfcc8/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-27_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_1.17.00.png)

- 비슷한 수준까지 올리긴 하나, LoRA 하이퍼파라미터만으로는 16비트 모델의 성능을 완전히 복제하지 못함
- 더 나은 성능을 위해 각 모델의 설정을 세밀하게 조정해야 함

🔎 **4-bit NormalFloat yields better performance than 4-bit Floating Point**

4비트 NormalFloat는 4비트 부동소수점보다 더 나은 성능을 제공

![스크린샷 2024-09-27 오후 1.17.13.png](QLoRA%20%E2%AD%90%2010d86814722780099048c1b2abfdfcc8/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-27_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_1.17.13.png)

![PPL이 낮을수록 좋음](QLoRA%20%E2%AD%90%2010d86814722780099048c1b2abfdfcc8/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-27_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_1.17.39.png)

PPL이 낮을수록 좋음

- LLaMA, OPT, BLOOM, Pythia와 같은 모델에서 Logistic Regression Modeling, Zero-shot Task에서 평가 진행
- 양자화된 데이터의 분포를 더 잘 보존할 수 있는 4bit NormalFloat가 더 좋은 성능을 보임
    - NormalFloat 방식은 정보 이론적으로 최적화된 양자화 방식
    - 데이터 분포의 백분위를 기준으로 양자화 구간을 나누기 때문에, FP4나 Int4보다 더 정확하게 값을 표현

🔎 **k-bit QLORA matches 16-bit full finetuning and 16-bit LoRA performance**

QLoRA 4비트 양자화를 사용하면서도 16비트 미세 조정과 거의 동등한 성능을 발휘

![T5와 RoBERTa 를 각각 비교해 Full finetuning 방식에 거의 유사하게 도달](QLoRA%20%E2%AD%90%2010d86814722780099048c1b2abfdfcc8/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-27_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_1.17.27.png)

T5와 RoBERTa 를 각각 비교해 Full finetuning 방식에 거의 유사하게 도달

![더 큰 크기의 모델로 적용할수록 QLoRA가 성능 향상 측면에서 좋다고 볼 수 있음](QLoRA%20%E2%AD%90%2010d86814722780099048c1b2abfdfcc8/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-27_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_1.17.55.png)

더 큰 크기의 모델로 적용할수록 QLoRA가 성능 향상 측면에서 좋다고 볼 수 있음

- 데이터셋 : FLAN v2, Alpaca
- LLaMA 7B ~ 65B 모델까지 fine-tuning한 후, [MMLU](https://discuss.pytorch.kr/t/mmlu/4064) benchmark(5-shot 평가)를 통해 성능을 측정
- **NF4, Double Quantization**를 결합하면 FP4보다 성능이 뛰어남
- 16비트 미세 조정과 거의 동일한 결과를 얻을 수 있음

### Pushing the Chatbot State-of-the-art with QLoRA

가장 큰 오픈소스 언어 모델들에 대해 instruction finetuning을 심층적으로 연구

instruction finetuing 성능평가를 위해 어려운 자연어 이해 벤치마크(MMLU)를 사용하여 평가를 수행

**Experimental setup**

- **Data**
    - instruction-following datasets은 존재하지 않아 타 데이터셋 8개 선택
    - OASST1 / 명령어-튜닝된 모델(Alpaca, Self-Instruct 등)에서의 추출된 데이터셋
        
        대규모 데이터 집합(FLAN v2 등) 등등 
        
    - 해당 데이터셋은 다양한 언어, 데이터 크기, 라이센스 범위를 포함
- **Training Setup**
    - 훈련 목표가 다른 데이터셋의 영향을 피하기 위해, QLoRA finetuing을 crossentropy로 수행
    - 모든 하이퍼파라미터 설정을 동일하게 유지한 상태에서 각 실험별 상위 응답을 선택
    - NF4 QLoRA와 이중 양자화를 결합하여 13B 및 33B LLaMA 모델에서 실험을 수행
- **Baselines**
    - commercial model(GPT-4, GPT-3.5-turbo, Bard)
    - opensource model(Vicuna, Open Assistant)
    
    ⇒ 다른 계열의 모델을 비교하며 진행
    

<aside>
🪜

**Evaluation**

QLoRA로 파인튜닝된 챗봇 성능을 평가하는 방법으로 MMLU와 생성 모델의 언어 능력을 평가하는 다양한 방식이 포함되어 있음. 이에 자동화된 평가와 인간 평가를 병행하여 성능을 측정

---

**MMLU Benchmark** Massive Multitask Language Understanding 

- 57개의 서로 다른 작업으로 구성된 **다중 선택형 테스트**
- 초등 수학, 미국 역사, 컴퓨터 과학, 법학 등 광범위한 주제를 다루고 있음

---

🔽 생성된 텍스트의 질을 평가를 위해 두가지 방법 사용

1. **Automated Evaluation**
    
    자동화된 평가에서는 GPT-4가 사용
    
    - GPT-4는 주어진 질문에 대해 ChatGPT-3.5 Turbo와 비교하는 방식으로 모델 응답을 평가
    - GPT-4는 ChatGPT와 비교
        - 각 응답에 10점 만점으로 점수를 부여하며, 두 응답을 비교하고, 상대적인 성능을 평가
        - 응답 순서가 점수에 영향을 미치는 것을 방지하기 위해, 두 가지 순서로 평가를 진행해 평균 점수를 계산
            
            ← 첫 번째 응답으로 모델 응답이 나왔을 때와, 두 번째로 나왔을 때 각각의 영향을 제거
            
    - 직접 비교 평가를 위해 **세가지 라벨링 시스템** 사용하며 Vicuna, OA 벤치마크 모두에서 적용
        1. 모델 간의 응답 중 하나가 더 나은 경우 채택
        2. 동점인 경우
        3. 설명을 제공해 모델 평가하는 방식
    - 결과
        - ChatGPT가 받은 점수 대비 모델의 응답이 몇 퍼센트나 되는지로 성능을 측정
    - ex. 모델이 ChatGPT보다 더 나은 응답을 제공하면, **100% 이상의 점수**를 받을 수 있음

1. **Human Evaluation**
    
    Amazon Mechanical Turk(AMT)에서 GPT-4와 비교하여 모델의 성능을 평가
    
    - 두 명의 인간 평가자가 각 모델 응답을 평가
    - GPT-4와 ChatGPT를 비교했으며, 이를 통해 모델의 성능을 더 정확하게 판단

---

**Elo Rating**

Elo 등급 시스템은 토너먼트 형식으로 모델들을 평가하는 것이며, Elo 등급은 체스나 다른 경쟁 게임에서 사용되는 평가 방법으로, 상대적인 승률을 기반으로 등급을 매김

- 두 모델 간 매치를 설정하여 가장 나은 응답을 제공한 모델이 승리
    - 승률에 따라 Elo 점수 부여
    - 예상치 못한 승리가 발생해 큰 점수 차가 나고 예상된 결과가 나오면 작은 점수 차이가 남
    - 시간 지남에 따라 모든 모델이 게임에서의 실력 수준에 맞춰 득점
- 실험 설정
    - 1000점을 초기 점수로 시작하
    - 32개의 모델이 서로 다른 순서로 10,000번의 시뮬레이션을 통해 비교

⇒ 모델의 **응답 순서**가 승패에 영향을 미치지 않도록 통제

</aside>

**Guanaco: QLORA trained on OASST1 is a State-of-the-art Chatbot**

> **Guanaco 모델을 개발해 성능 증명**
> 
> 
> QLoRA로 finetuing된 모델이며 좋은 성능을 내고 적은 메모리로도 뛰어난 성능을 발휘할 수 있음
> 
1. **Automated Evaluation**
    
    ![image.png](QLoRA%20%E2%AD%90%2010d86814722780099048c1b2abfdfcc8/image%201.png)
    
    - Chat-GPT 모델을 BASE로 설정하며 이에 대비해 얼마나 좋은 성능을 보이는 지 검증
    - {GPT모델의 답 : Guanaco모델의 답}을 GPT-4에 넣어 점수를 부여하도록 함
    - Mean(bold처리)된 값이 ChatGPT에 비해 우월한 성능이라 보면 됨
    - 훨씬 더 적은 메모리로도 좋은 성능을 보이고 있음

1. **Human Evaluation**
    
    Guanaco가 ChatGPT보다 우수한 성능을 보이고 있음
    
    ![image.png](QLoRA%20%E2%AD%90%2010d86814722780099048c1b2abfdfcc8/image%202.png)
    

### ~~Qualitative Analysis~~

## 4. Conclusion

---

### Contribution

- LLM 파인튜닝은 많은 GPU 메모리 소요
    
    ⇒ LoRA에 양자화를 적용해 비용 절감
    
- 양자화 시 정확도 손실 줄일 필요 있음
    
    ⇒ NormalFloat이라는 새로운 데이터 타입 제시
    
    ⇒ LoRA Adapter를 모든 가중치에 적용하여 양자화 에러를 최소화하도록 함
    

### Future work

- NF4 데이터 타입은 custom 데이터 타입이기에 오버헤드가 발생할 수 있음
- 페이지 옵티마이저.. 도 퓨처워크
- QDoRA…

## 🎹 코드 리뷰

---

## ➕ 참고

---

[[Ambient AI] Student Presentation - QLoRA](https://www.youtube.com/watch?v=0jsOaPhoQXs)

[인공지능의 성적표 - MMLU에 대해 알아봅시다](https://aibear.eeeyap.com/16)