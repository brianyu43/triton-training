# Reinforcement Learning System Plan

작성일: 2026-04-29 KST

## 판단

NanoTriton-LM에 강화학습 체계를 붙이는 것은 가치가 있다. 다만 처음부터 RLHF나 대형 instruct tuning을 흉내 내는 방향은 맞지 않다. 이 프로젝트의 장점은 작은 모델, 직접 만든 학습 루프, Triton kernel, VM 검증을 끝까지 연결하는 데 있다.

따라서 강화학습 축도 같은 원칙으로 간다.

목표는 “작은 모델이 reward를 따라 policy를 업데이트하고, 그 과정의 logprob / KL / advantage / rollout / eval을 우리가 직접 확인할 수 있는 체계”를 만드는 것이다.

## 왜 하는가

강화학습 체계는 다음 역량을 보여준다.

- autoregressive generation을 학습 loop 안에서 다룬다.
- token logprob와 sequence reward를 정확히 계산한다.
- supervised loss와 policy-gradient 계열 loss의 차이를 이해한다.
- reward hacking, KL collapse, entropy 감소 같은 실패 모드를 관찰한다.
- 나중에 Triton으로 최적화할 후보가 생긴다.
  - logprob gather
  - KL penalty
  - advantage normalization
  - masked reductions
  - sequence-level reward aggregation

즉, RL은 “모델을 더 똑똑하게 만드는 장식”이 아니라, training system을 더 깊게 만드는 실험 축이다.

## 하지 않을 것

초기에는 다음을 하지 않는다.

- 인간 preference collection
- 큰 instruction dataset
- reward model pretraining
- multi-GPU RL
- production-grade RLHF framework 흉내
- 외부 framework를 그대로 가져와서 black box로 돌리기

이것들은 나중에 reference로 볼 수는 있지만, 프로젝트의 첫 RL 단계에서는 오히려 학습 목적을 흐린다.

## 권장 설계

RL 체계는 세 층으로 나눈다.

```text
Level 1. Toy RL correctness
  작은 prompt와 deterministic reward
  policy update가 의도대로 움직이는지 확인

Level 2. Language-shaping RL
  Tiny Shakespeare 또는 synthetic text에서 형식/스타일 reward
  KL penalty로 baseline language model에서 너무 멀어지지 않게 제어

Level 3. Kernel-aware RL training system
  rollout, logprob, KL, reward aggregation 중 병목을 찾고 Triton 후보로 분리
```

## Level 1. Toy RL correctness

목표: 강화학습 수식과 구현이 맞는지 작은 문제에서 검증한다.

추천 task:

- prompt 뒤에 특정 문자 나오면 reward +1
- 괄호를 닫으면 reward +1
- 숫자 prompt 뒤에 짝수/홀수 token을 맞추면 reward +1
- 짧은 sequence에서 forbidden token을 피하면 reward +1

중요한 점은 reward가 자동 계산 가능해야 한다는 것이다. 사람이 평가하는 reward는 이 단계에서 쓰지 않는다.

산출물 후보:

```text
nanotriton/rl/
  __init__.py
  rollout.py
  rewards.py
  losses.py
  trainer.py

tests/
  test_rl_rewards.py
  test_rl_logprobs.py
  test_rl_loss.py
```

성공 기준:

- fixed seed에서 rollout이 재현된다.
- generated token의 logprob가 PyTorch reference와 일치한다.
- reward 함수가 작은 hand-written cases를 통과한다.
- loss backward가 NaN 없이 돈다.
- 몇십 step 안에 reward 평균이 올라간다.

## Level 2. Language-shaping RL

목표: 실제 language model checkpoint 위에서 reward를 주고 policy를 움직인다.

초기 reward는 단순해야 한다.

예시:

- 줄바꿈 포함 reward
- 특정 character set 유지 reward
- 너무 짧거나 너무 긴 generation penalty
- repeated character penalty
- prompt와 같은 style의 punctuation 사용 reward

이 단계에서 핵심은 KL penalty다.

```text
objective = reward - beta * KL(policy || reference_policy)
```

reference_policy는 Milestone 1의 PyTorch baseline checkpoint를 사용한다. 이렇게 해야 reward 하나만 보고 모델이 language model 성질을 잃는 것을 막을 수 있다.

성공 기준:

- reward 평균이 올라간다.
- KL이 폭주하지 않는다.
- 생성 샘플이 완전히 망가지지 않는다.
- 같은 prompt set으로 before/after sample을 비교할 수 있다.

## Level 3. Kernel-aware RL

목표: RL loop 안에서 Triton으로 최적화할 연산 후보를 찾는다.

처음부터 RL optimizer를 Triton으로 만들 필요는 없다. 대신 다음 관측을 한다.

- rollout 시간
- forward logprob 계산 시간
- KL 계산 시간
- masked reduction 시간
- reward aggregation 시간
- backward 시간

후보 kernel:

```text
masked_mean
masked_sum
logprob_gather
kl_divergence_tokens
advantage_normalize
```

이 단계가 되면 강화학습 체계도 NanoTriton-LM의 kernel roadmap과 자연스럽게 만난다.

## Algorithm Path

처음에는 가장 단순한 순서로 간다.

```text
1. REINFORCE-style sequence reward
2. baseline-normalized policy gradient
3. KL-regularized policy gradient
4. PPO-style clipped objective
5. group-relative advantage variant
```

초기 구현은 1~3까지만 해도 충분하다. PPO 계열은 logging과 stability가 필요하므로, rollout/logprob/reward가 믿을 수 있게 된 뒤에 간다.

## 데이터 흐름

```text
prompts
  -> policy.generate(...)
  -> generated sequences
  -> logprobs under policy
  -> logprobs under reference policy
  -> reward function
  -> KL penalty
  -> advantage
  -> policy loss
  -> optimizer step
```

이 흐름은 반드시 작은 batch에서 중간 tensor를 저장하고 inspect할 수 있게 만든다.

## Metrics

최소 logging:

- mean reward
- mean KL
- mean entropy
- mean response length
- policy loss
- approximate gradient norm
- samples before/after

저장 위치:

```text
out/rl_toy/
  metrics.jsonl
  samples_step_0000.txt
  samples_step_0100.txt
  checkpoint.pt
```

## 외부 Reference 원칙

강화학습 관련 외부 repository나 model을 문서에 구체적으로 언급하게 되면, 기존 원칙대로 실제로 fetch하고 commit hash를 pin한다.

초기 RL 설계 문서에서는 특정 GitHub repository를 고정하지 않는다. 먼저 우리 손으로 작은 loop를 만든다.

## 권장 Milestone

### Milestone RL-0. Logprob and Reward Harness

목표: 학습 없이 RL에 필요한 측정 도구를 만든다.

산출물:

- rollout helper
- generated token logprob 계산
- reward function registry
- prompt fixture
- before/after sample writer

성공 기준:

- fixed prompt에서 logprob shape가 정확하다.
- reward 함수 unit test가 통과한다.
- reference policy와 policy의 KL을 계산할 수 있다.

### Milestone RL-1. Toy Policy Gradient

목표: deterministic reward toy task에서 reward 평균이 올라가는지 확인한다.

산출물:

- sequence-level policy gradient loss
- advantage normalization
- RL trainer
- `configs/rl_toy.yaml`

성공 기준:

- L4에서 100~500 step 안에 mean reward가 상승한다.
- KL과 entropy를 같이 기록한다.
- checkpoint reload 후 sample generation이 된다.

### Milestone RL-2. KL-Regularized Language RL

목표: Tiny Shakespeare checkpoint 위에서 단순 style reward를 적용한다.

성공 기준:

- reward 평균이 상승한다.
- KL이 설정한 범위 안에서 유지된다.
- sample이 reward만 과최적화한 이상한 문자열로 붕괴하지 않는다.

## 지금 당장 다음 작업으로 넣을지

바로 구현하기보다는 RMSNorm backward와 model integration을 먼저 끝내는 편이 좋다.

추천 순서:

```text
1. RMSNorm backward + autograd wrapper
2. model에서 RMSNorm만 Triton module로 교체
3. baseline vs Triton RMSNorm loss curve regression
4. RL-0 logprob/reward harness
5. RL-1 toy policy gradient
```

이 순서가 좋은 이유는 RL이 model forward/generation/logprob에 강하게 의존하기 때문이다. 모델 교체 체계가 안정된 뒤 RL을 올려야 디버깅이 쉬워진다.

## 결론

강화학습 체계는 넣자. 다만 프로젝트의 성격에 맞게 “작고 검증 가능한 RL training system”으로 시작한다.

첫 구현 목표는 RLHF가 아니라 다음 문장이다.

> 작은 GPT가 자동 reward를 받고, KL을 보면서, policy update로 평균 reward를 올리는 것을 end-to-end로 증명한다.

이것이 되면 NanoTriton-LM은 kernel project에서 training system project로 한 단계 넓어진다.
