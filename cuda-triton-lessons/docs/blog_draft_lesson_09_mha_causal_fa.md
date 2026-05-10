# 300 줄짜리 Triton Flash Attention 이 cuDNN FA-2 의 80-90 % 를 따라잡는 방법

*LLaMA-7B shape 기준, L4 sm_89 에서. `torch.compile(fullgraph=True)` 호환까지.*

기준 날짜: 2026-04-20  ·  GPU: NVIDIA L4 (Ada Lovelace, 24GB)  ·  CUDA 13.0 · PyTorch 2.11 · Triton 3.6

---

레슨 8 에서 2-D (`N, d`) Triton Flash Attention 을 썼다. ~100 줄, non-causal forward. L4 에서 우리 CUDA FA v1 대비 6.1× 빨랐고 cuDNN FA-2 의 79 % 속도였다.

그런데 실제 LLM 의 attention 은 그 shape 가 아니다. 실제 shape 는 `(batch, heads, seq_len, head_dim)` — 4-D. 그리고 대부분 **causal** (decoder). 그리고 `torch.compile` 아래에서 돌아야 쓸모가 있다.

이번 글은 그 간격을 100 줄 → 300 줄 로 메꾸는 과정이다:

1. 2-D → 4-D 확장 (stride 네 개, 3-D grid)
2. `IS_CAUSAL: tl.constexpr` + FA-v2 loop-skip
3. `torch.library.custom_op` 로 `torch.compile(fullgraph=True)` 통과

결과: **LLaMA-7B-like causal shape 에서 cuDNN-backed `F.scaled_dot_product_attention` 의 78-90 % 속도**, GPT-2-like 짧은 shape 에서 **동률 또는 13 % 우세**, naive 대비 29-33×. `torch.compile` 이 그래프를 안 끊음.

## 1. 실험 설계

같은 L4 VM 에서 세 구현을 같은 `(B, H, N, d)` 에 돌림.

1. **Ours** — `triton_flash_attention_mha` (이 레슨에서 확장한 Triton 커널)
2. **SDPA** — `F.scaled_dot_product_attention` — 토치가 backend 고름, L4 + fp16 에서 cuDNN FA-2 경로
3. **Naive** — `(Q @ K^T * scale).softmax(-1) @ V`, causal 이면 하삼각 mask

벤치: `triton.testing.do_bench(fn, warmup=25, rep=100)` — median 을 잰다. FLOP 은 `4 × B × H × N² × d`, causal 이면 절반.

관측: fp16, fp32 reference 대비 max rel_err 는 `3 × 10⁻⁴` 수준 — 수치적으론 bit-perfect 에 가깝다.

## 2. 결과 — LLaMA-7B causal, d=128

| (B, H, N)     | ours                | SDPA (FA-2) | ours/SDPA | vs naive |
|---------------|---------------------|-------------|-----------|----------|
| (1, 32, 512)  | 0.100 ms (21.5 TF)  | 0.100 ms    | **1.00×** | 10.97×   |
| (1, 32, 1024) | 0.223 ms (38.5 TF)  | 0.202 ms    | 0.90×     | 29.6×    |
| (1, 32, 2048) | 0.784 ms (43.8 TF)  | 0.613 ms    | 0.78×     | 31.4×    |
| (1, 32, 4096) | 2.964 ms (46.4 TF)  | 2.559 ms    | 0.86×     | 32.7×    |
| (2, 32, 1024) | 0.428 ms (40.1 TF)  | 0.373 ms    | 0.87×     | 30.6×    |
| (2, 32, 2048) | 1.565 ms (43.9 TF)  | 1.372 ms    | 0.88×     | 31.3×    |
| (4, 32, 1024) | 0.886 ms (38.8 TF)  | 0.726 ms    | 0.82×     | 29.3×    |

GPT-2 shape, d=64 causal:

| (B, H, N)     | ours                | SDPA       | ours/SDPA |
|---------------|---------------------|------------|-----------|
| (8, 12, 1024) | 0.302 ms (42.7 TF)  | 0.303 ms   | **1.00×** |
| (16, 12, 512) | 0.249 ms (25.9 TF)  | 0.282 ms   | **1.13×** |

**요약**:
- d=128 에서 FA-2 의 78-90 %
- d=64 에선 동률 / 13 % 우세
- naïve 는 compute 에 안 들어갈 만한 느림 (29-33×)
- 핵심 kernel body 약 100 줄, op 래퍼 76 줄 포함해도 300 줄 미만

## 3. 4-D 확장 — stride 네 개, grid 세 개

레슨 8 의 2-D kernel 이 이렇게 생겼었다 (핵심만):

```python
@triton.jit
def flash_attention_2d_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qn, stride_qk,
    stride_kn, stride_kk,
    ...
    N, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    # Q pointer: q_base + offs_m[:, None]*stride_qn + offs_d[None, :]*stride_qk
```

`(B, H, N, d)` 4-D 로 올리면 각 텐서 당 stride 가 4 개 (`stride_qb, stride_qh, stride_qm, stride_qk`). 그리고 `(B, H)` 짝마다 완전히 독립적인 attention 문제 — 이걸 grid 차원으로 표현하는 게 자연스럽다:

```python
grid = (triton.cdiv(N, BLOCK_M), H, B)
```

Grid 3-D 를 받은 kernel 은:

```python
pid_m = tl.program_id(axis=0)   # BLOCK_M of queries
pid_h = tl.program_id(axis=1)   # which head
pid_b = tl.program_id(axis=2)   # which batch element

# 한 번만 base 를 shift. 이 네 줄 아래부터는 2-D 코드와 동일.
q_base = Q_ptr   + pid_b * stride_qb + pid_h * stride_qh
k_base = K_ptr   + pid_b * stride_kb + pid_h * stride_kh
v_base = V_ptr   + pid_b * stride_vb + pid_h * stride_vh
o_base = Out_ptr + pid_b * stride_ob + pid_h * stride_oh
```

**포인트**: `(batch, head)` 를 **루프로** 안 돌리고 **grid 로** 돌린다는 것. Triton 런치는 이 grid 전체를 "available 하면 어느 program 부터든 동시에 시작" 시키므로, L4 의 58 SM 이 최대한 채워진다. 루프로 풀었다면 SM 들이 놀았을 것.

대부분 LLM shape 에서 `B × H` 가 `BLOCK_M` 기반 tile 수만큼 있어서, 이 3-D grid 로도 충분히 occupancy 가 꽉 참.

## 4. Causal 의 진짜 속도는 mask 가 아니다 — loop skip 이다

Causal attention 의 "상식" 구현:

```python
s = q @ k.T * scale
mask = torch.ones(N, N).tril()
s = s.masked_fill(~mask, float("-inf"))
p = s.softmax(-1)
```

이건 **속도 이득이 0**. Mask 를 쓰든 말든 K/V 전체를 다 읽고 전체 `N × N` 스코어를 다 계산한다. 단지 결과 값이 `-inf` 인 것뿐.

FA-v2 의 causal 최적화는 한 줄로 요약된다:

> 상삼각에 **완전히** 들어가는 K 타일은 **이터레이션 자체에서 뺀다**.

우리 커널로 옮기면:

```python
if IS_CAUSAL:
    end_n = tl.minimum(N, (pid_m + 1) * BLOCK_M)
else:
    end_n = N

for start_n in range(0, end_n, BLOCK_N):
    # load K/V tile, tl.dot, online softmax merge...
```

이 `pid_m` 이 다루는 Q 블록의 **마지막 행** 이 `(pid_m + 1) * BLOCK_M - 1`. 그것보다 column index 가 큰 K 타일은 전부 상삼각이라 **읽을 가치가 없다**. 그래서 이터레이션 상한을 잘라버린다.

평균 `N/2` 개 타일만 돌고 끝. K/V 로딩, `tl.dot` 두 번, online softmax merge — 전부 반으로 줄어든다.

그리고 대각선을 걸치는 타일 하나 — 거기만 mask 가 실질적으로 적용된다:

```python
if IS_CAUSAL:
    causal_mask = offs_m[:, None] >= offs_n[None, :]
    s = tl.where(causal_mask, s, -float("inf"))
```

**검증**: `(1, 32, 2048, 128)` 에서 non-causal 2.643 ms, causal 0.784 ms. FLOP 은 절반인데 시간은 그보다 더 떨어졌다 (3.3×). 이유: 타일 수가 줄면 pipeline startup 이 한 번만 더 amortize 된다. 덤.

## 5. `tl.constexpr` — 런타임 if 를 컴파일 타임으로 접기

위의 `IS_CAUSAL` 분기는 kernel body 의 **핫 루프 안** 에 있다. 그냥 파이썬 bool 로 넘기면:
- 매 iter 에 `if is_causal:` 평가
- PTX 레벨에서 branch + predicated execution
- warp 가 발 묶임

해결: `tl.constexpr` 로 찍는다.

```python
@triton.jit
def flash_attention_mha_fwd_kernel(
    ...,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
```

그리고 런치 쪽:

```python
flash_attention_mha_fwd_kernel[grid](
    ...
    HEAD_DIM=head_dim,
    IS_CAUSAL=is_causal,
)
```

이러면 Triton 이 `IS_CAUSAL=True` 용 커널과 `=False` 용 커널 **두 개를 별도로 JIT 컴파일**. 런타임엔 해당하는 하나만 dispatch. 핫 루프에 if 자체가 없다.

그리고 **autotune 의 key 에도 포함**시키면:

```python
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N", "HEAD_DIM", "IS_CAUSAL"])
```

causal / non-causal 별도로 tile size 를 튜닝한다. 실제로 causal 은 `BLOCK_M=64, BLOCK_N=128`, non-causal 은 `BLOCK_M=128, BLOCK_N=64` 가 더 좋더라 — 루프 제한이 다르니 최적 shape 이 다를 수밖에. 하나의 kernel 이 두 전략을 모두 가질 수 있게 됐다.

## 6. `torch.library.custom_op` — "드롭인 커널" 로 올리기

여기까지만 해도 커널은 Python 에서 직접 부를 수 있다:

```python
out = triton_flash_attention_mha(q, k, v, is_causal=True)
```

근데 이건 두 가지 환경에서 문제가 된다.

### 6.1 `torch.compile` 에서 그래프가 끊긴다

`torch.compile(model, fullgraph=True)` 은 **미지의 Python 함수** 를 만나면 그래프를 끊는다 (graph break). 우리 커널 위치에서:

```python
def forward(x):
    qkv = linear_qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)
    out = triton_flash_attention_mha(q, k, v, is_causal=True)   # ← 여기서 break
    return linear_out(out)
```

Dynamo 입장: 이 함수가 뭘 반환하는지 모름 (shape? dtype? device?). 안전을 위해 여기서 trace 를 끊고 실제 eager 실행에 맡긴다. `fullgraph=True` 면 에러.

### 6.2 `torch.export` 에 이름이 안 남는다

ONNX / TorchScript 로 내보낼 때 이 커널은 단지 "파이썬 함수 호출" 이라서 그래프에 이름으로 남지 않는다.

### 해결: `@custom_op` 로 "공식 op" 로 등록

```python
from torch.library import custom_op

@custom_op(
    "triton_training::flash_attention_mha",
    mutates_args=(),               # pure function, 입력 변경 없음
    device_types="cuda",           # CUDA only; CPU 면 명확한 에러
)
def flash_attention_mha_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    return triton_flash_attention_mha(q, k, v, is_causal=is_causal)


@flash_attention_mha_op.register_fake
def _flash_attention_mha_fake(q, k, v, is_causal=False):
    # FakeTensor (no data) 에서 호출됨. shape/dtype/device 만 선언.
    return torch.empty_like(q)
```

이제 세 경로가 모두 쓸 수 있다:

```python
# 1) Python 으로 직접
out = flash_attention_mha_op(q, k, v, True)

# 2) torch.ops path (vLLM 스타일)
out = torch.ops.triton_training.flash_attention_mha(q, k, v, True)

# 3) torch.compile 아래
compiled = torch.compile(model, fullgraph=True)
out = compiled(x)   # 안에서 op 를 부르는 코드가 있어도 graph break 없음
```

세 경로가 **bit-exact** 로 같은 값을 낸다 (검증: `err == 0.00e+00` × 4 케이스).

### 6.3 `fullgraph=True` 로 LLaMA-style attention block 전체 통과

검증 테스트 — 진짜 attention block 하나를 `torch.compile(..., fullgraph=True)` 에 통째로 넣는다:

```python
class AttentionBlock(nn.Module):
    def __init__(self, d_model=1024, n_heads=16):
        super().__init__()
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        out = torch.ops.triton_training.flash_attention_mha(q, k, v, True)
        return self.out(out.transpose(1, 2).reshape(B, N, D))


model = AttentionBlock().cuda().half()
compiled = torch.compile(model, fullgraph=True)
y = compiled(x)   # ← 그래프 브레이크 없이 전체가 한 그래프
```

결과: `eager vs compiled err = 0.00e+00`, `fullgraph=True` 통과.

**이게 vLLM 이 커스텀 커널을 제공하는 패턴** 과 정확히 같다. vLLM 의 `torch.ops.vllm.*` 가 그런 식으로 옷을 입고 있고, 덕분에 `torch.compile` 과 호환된다.

## 7. 아직 남은 20 %

LLaMA shape 에서 SDPA 의 78-90 %. 갭이 왜 있는가?

SDPA 가 L4 + fp16 에서 호출하는 건 cuDNN 의 FA-2 구현이다. 그 안엔:

1. **Async copy + 멀티 버퍼** — smem 에 K/V 로드 하는 동안 이전 타일 `mma` 진행. Triton 의 `num_stages` 로 일부 흉내 내지만 cuDNN 만큼 세밀 제어는 아님.
2. **Persistent kernel 스케줄링** — 블록을 계속 살려두고 타일을 재분배. 매 타일마다 launch overhead 지불 안 함.
3. **Warp specialization** — 일부 warp 는 compute, 일부는 load/store 전담. Triton 에도 오는 중이지만 현재 main 브랜치는 아직 experimental.

다음 방향 셋:

- **Backward + autograd** (훈련에 쓰이려면 필수)
- **GQA 지원** (LLaMA-2/3 은 K/V head 수가 Q 의 1/4)
- **Persistent + async copy** (나머지 20 % 갭 닫기)

이 셋은 각각 독립적으로 의미 있고, 셋 다 해놓으면 vLLM PagedAttention 포팅 (내 Phase 1 M3 목표) 의 출발점이 된다.

## 8. 레슨의 요지 — 300 줄이 하는 일

[triton_kernels/flash_attention_mha.py](../triton_kernels/flash_attention_mha.py) — 192 줄 (docstring + autotune configs 포함, kernel body ~100 줄)
[triton_kernels/flash_attention_mha_op.py](../triton_kernels/flash_attention_mha_op.py) — 76 줄

합쳐서 268 줄.

이 268 줄이:

| Shape | 우리 Triton | SDPA (cuDNN FA-2) | 비율 |
|---|---|---|---|
| LLaMA-7B, N=4096, causal | 2.964 ms | 2.559 ms | 0.86× |
| LLaMA-7B, N=2048, B=2, causal | 1.565 ms | 1.372 ms | 0.88× |
| GPT-2, N=512, B=16, causal | 0.249 ms | 0.282 ms | **1.13×** |

그리고:

- naïve attention 대비 29-33×
- `torch.compile(fullgraph=True)` 에서 그래프 브레이크 0 건
- `torch.ops.triton_training.flash_attention_mha` 로 드롭인 가능

**Triton 의 ROI** 는 이 경계에 있다 — "cuDNN 의 80-90 % 를 한 파일에서, 새 GPU 재컴파일이 `@triton.autotune` 한 줄로 처리되는." cuDNN 의 마지막 10-20 % 를 좇는 건 다른 전투다 (persistent + async copy + warp specialization, 아직은 CUTLASS/CUDA 의 세계).

---

## Appendix: 세 구현 요약

우리:
```python
@triton.autotune(configs=..., key=["N", "HEAD_DIM", "IS_CAUSAL"])
@triton.jit
def flash_attention_mha_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    ...,
    N, scale,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m, pid_h, pid_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    ...
    end_n = tl.minimum(N, (pid_m + 1) * BLOCK_M) if IS_CAUSAL else N
    for start_n in range(0, end_n, BLOCK_N):
        k = tl.load(k_ptrs); v = tl.load(v_ptrs)
        s = tl.dot(q, tl.trans(k)) * scale
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, -inf)
        # online softmax + acc update
    tl.store(out_ptrs, acc / l_i[:, None])
```

Op 래퍼:
```python
@custom_op("triton_training::flash_attention_mha", mutates_args=(), device_types="cuda")
def flash_attention_mha_op(q, k, v, is_causal=False) -> torch.Tensor:
    return triton_flash_attention_mha(q, k, v, is_causal=is_causal)

@flash_attention_mha_op.register_fake
def _fake(q, k, v, is_causal=False):
    return torch.empty_like(q)
```

쓰는 쪽:
```python
import triton_kernels.flash_attention_mha_op   # side-effect import

out = torch.ops.triton_training.flash_attention_mha(q, k, v, is_causal=True)
# torch.compile(fullgraph=True) 와 호환
```

전체 재현: [`github.com/brianyu43/triton-training`](https://github.com/brianyu43/triton-training) · 레슨 09.
