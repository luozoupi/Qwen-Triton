# Qwen-Triton

Qwen-Triton is a repo-owned Qwen3/Qwen3.5 bring-up where the dense Qwen3 inference and training path is **fully implemented in Triton** -- every operator from embedding lookup through cross-entropy loss runs through a repo-owned Triton kernel on GPU.

The current validated target is text-only `Qwen/Qwen3-0.6B-Base`. Dense Qwen3 checkpoint load, forward, generation, and short Wikitext fine-tuning all run on GPU with zero `nn.Linear` or `nn.Embedding` modules remaining. Qwen3 MoE and Qwen3.5 text-family support are scaffolded in the config and module layers, but they have not yet been profiled and benchmarked as thoroughly as dense Qwen3.

## Current Status

- Validated model target: `Qwen/Qwen3-0.6B-Base`
- **Full Triton dense path**: all 197 linear projections use `TritonLinear`, embedding uses `TritonEmbedding`, attention uses Triton Flash Attention v2, loss uses Triton cross-entropy, residual connections use Triton element-wise add
- Strict Triton mode (`QWEN_TRITON_STRICT=1`): forward + backward pass with zero PyTorch fallbacks
- Working backends:
  - `ref`: upstream Hugging Face model for parity and baseline measurement
  - `triton`: repo-owned module stack with all Triton kernels
- Working RoPE backends:
  - default Triton RoPE path
  - explicit CUDA custom operator via `QWEN_TRITON_ROPE_BACKEND=cuda_op`
- Forward parity with HF reference: `logit_max_abs_diff ~1.1` (bf16)
- Training validated: 20-step WikiText fine-tuning, finite loss and gradients throughout

## Architecture Overview

### 1. Config Normalization

`qwen_triton.configs.QwenTritonConfig` normalizes Hugging Face configs into one internal representation with four families:

- `qwen3_dense`
- `qwen3_moe`
- `qwen35_text_dense`
- `qwen35_text_moe`

It accepts both plain Qwen3 configs and Qwen3.5-style `text_config` payloads. For Qwen3.5-family configs it also derives `layer_types`, so decoder layers can dispatch between:

- `full_attention`
- `sliding_attention`
- `linear_attention`

This normalization layer is what lets the rest of the code work from one stable internal config instead of branching on raw HF schema differences everywhere.

### 2. Model Stack

The public entrypoint is `QwenTritonForCausalLM`. Internally it builds:

- `TritonEmbedding` token embeddings
- a decoder stack of `QwenDecoderLayer`
- final Triton RMSNorm
- tied `TritonLinear` LM head when configured

Each decoder layer does:

1. Triton RMSNorm
2. Triton Flash Attention (with `TritonLinear` Q/K/V/O projections, Triton RoPE, Triton RMSNorm head norms)
3. Triton residual add
4. Triton RMSNorm
5. Triton MLP (`TritonLinear` gate/up/down projections + Triton SiLU-mul)
6. Triton residual add

Dense Qwen3 is the most validated path today.

### 3. Attention

`QwenFullAttention` is fully repo-owned Triton:

- Q/K/V/O projections use `TritonLinear` (Triton GEMM kernel)
- Q/K head normalization uses Triton RMSNorm
- Rotary embedding uses Triton RoPE (or optional CUDA custom op)
- KV cache update uses Triton write-by-position kernels
- Attention score computation uses **Triton Flash Attention v2** with:
  - Online softmax with causal masking computed inline (no mask tensor materialized)
  - Native GQA support (Q heads map to KV heads inside the kernel, no `repeat_kv`)
  - fp32 accumulation (inputs stay in bf16)
  - Forward + backward kernels with logsumexp saved for gradient recomputation
- Attention output gating uses Triton sigmoid-mul for gated-attention families
- Fallback to SDPA only for complex mask scenarios (e.g. sliding window with padding)

### 4. MLP / MoE / Qwen3.5

Dense MLP is fully Triton:

- `gate_proj` / `up_proj` / `down_proj` all use `TritonLinear`
- Fused SiLU-mul epilogue uses Triton kernel

MoE and Qwen3.5 linear-attention blocks are scaffolded so the repo can represent:

- dense Qwen3
- sparse Qwen3 MoE
- Qwen3.5 text blocks with mixed layer types
- shared-expert style Qwen3.5 text-family MoE

The dense Qwen3 path is the only path fully exercised end-to-end on a real checkpoint in this README.

### 5. Triton and CUDA Kernels

Complete repo-owned Triton kernel inventory:

| Kernel | File | Forward | Backward | Purpose |
|--------|------|:-------:|:--------:|---------|
| RMSNorm | `kernels/rmsnorm.py` | Triton | Triton | Layer normalization |
| SiLU-mul | `kernels/swiglu.py` | Triton | Triton | MLP gating (SwiGLU) |
| Sigmoid-mul | `kernels/sigmoid_mul.py` | Triton | Triton | Attention output gating |
| RoPE | `kernels/rope.py` | Triton/CUDA | Triton/CUDA | Rotary positional embeddings |
| KV Cache | `kernels/cache.py` | Triton | N/A | Write-by-position cache updates |
| Gated Delta Rule | `kernels/linear_attention.py` | Triton | Implicit | Qwen3.5 linear attention |
| **Matmul/GEMM** | `kernels/matmul.py` | Triton | Triton | All linear projections |
| **Flash Attention v2** | `kernels/flash_attention.py` | Triton | Triton | Scaled dot-product attention |
| **Softmax** | `kernels/softmax.py` | Triton | Triton | Standalone softmax (MoE routing) |
| **Embedding** | `kernels/embedding.py` | Triton | Triton | Token embedding lookup |
| **Cross-Entropy** | `kernels/cross_entropy.py` | Triton | Triton | Fused softmax + NLL loss |
| **Residual Add** | `kernels/residual_add.py` | Triton | Identity | Element-wise residual connection |

Explicit CUDA custom operator (optional):

- `qwen_triton/csrc/rope_op.cpp`
- `qwen_triton/csrc/rope_op_kernel.cu`
- Python wrapper in `qwen_triton/ops/rope_cuda.py`

The RoPE path is selectable by environment variable:

```bash
export QWEN_TRITON_ROPE_BACKEND=triton
export QWEN_TRITON_ROPE_BACKEND=cuda_op
export QWEN_TRITON_ROPE_BACKEND=auto
```

### 6. Module Layer

The module layer provides drop-in replacements for PyTorch primitives:

| Module | Replaces | Parameter Compatibility |
|--------|----------|----------------------|
| `TritonLinear` | `nn.Linear` | Same `weight`/`bias` names and shapes |
| `TritonEmbedding` | `nn.Embedding` | Same `weight` name and shape |
| `QwenRMSNorm` | Custom | Triton kernel with optional `one_plus_weight` |
| `QwenRMSNormGated` | Custom | Gated variant using sigmoid-mul |

`TritonLinear` and `TritonEmbedding` are designed so that `load_state_dict` and the HF weight loader work without any name mapping changes.

### 7. Loader Design

`from_pretrained_hf(...)` does three things:

1. normalize the Hugging Face config into `QwenTritonConfig`
2. build either the reference or repo-owned model
3. map HF safetensor names into repo-owned parameter names

The Triton path intentionally avoids reusing the upstream HF module tree. That keeps the public construction API stable while allowing kernel and module implementation to change underneath it.

## Code Layout

- `qwen_triton/configs`
  - internal config normalization and family detection
- `qwen_triton/models`
  - public model API and backend switch
- `qwen_triton/modules`
  - decoder blocks, attention, rotary, MLP, MoE, cache, TritonLinear, TritonEmbedding
- `qwen_triton/kernels`
  - Triton kernels and their runtime wrappers (12 kernel files)
- `qwen_triton/ops`
  - Python loader/wrapper for custom CUDA operators
- `qwen_triton/csrc`
  - C++/CUDA extension code for the RoPE custom op
- `qwen_triton/loaders`
  - Hugging Face snapshot/config/safetensor loading and weight mapping
- `qwen_triton/scripts`
  - smoke, train, benchmark, profile, and CUDA-op build scripts
- `tests`
  - kernel parity tests and GPU regression coverage

## Correctness Notes

The dense Qwen3 path is now fully Triton with verified correctness:

- All 12 Triton kernels have forward parity with their PyTorch equivalents
- Backward kernels (where applicable) produce matching gradients
- Forward logit parity with HF reference model: `max_abs_diff ~1.1` in bf16
- 20-step WikiText training runs to completion with finite loss and gradients
- Model parity tests pass for dense Qwen3 (tiny configs)
- Strict Triton mode (`QWEN_TRITON_STRICT=1`) enforces zero fallbacks

Historical note: the original Triton primitives for RMSNorm, SiLU-mul, and RoPE were forward-only and therefore detached from autograd. That was fixed by adding Triton backward kernels for all three. The full-Triton work then extended this pattern to all remaining operators (GEMM, attention, embedding, cross-entropy, residual add).

## Environment

Create a fresh conda environment from the repo root with:

```bash
cd /home/luo00466/Qwen-Triton
conda env create -f environment.yml
conda activate qwen-triton
```

If you already have a Python 3.10 environment and only want the pip packages, install:

```bash
python -m pip install -r requirements.txt
```

Build the CUDA RoPE operator after the environment is active with:

```bash
TORCH_CUDA_ARCH_LIST=12.0 python -m qwen_triton.scripts.build_rope_cuda_op --verbose
```

## Quick Start

### Smoke Test

```bash
CUDA_VISIBLE_DEVICES=3 python -m qwen_triton.scripts.smoke \
  --model-id Qwen/Qwen3-0.6B-Base \
  --backend triton \
  --device cuda \
  --dtype bf16 \
  --max-new-tokens 4 \
  --compare-ref \
  --strict-triton
```

### One-Step Wikitext Train/Eval Smoke

```bash
CUDA_VISIBLE_DEVICES=3 python -m qwen_triton.scripts.train_wikitext \
  --model-id Qwen/Qwen3-0.6B-Base \
  --backend triton \
  --dataset wikitext-2-raw-v1 \
  --device cuda \
  --dtype bf16 \
  --seq-len 128 \
  --train-steps 20 \
  --eval-batches 5 \
  --lr 1e-5 \
  --strict-triton
```

### Run with Strict Triton Mode

Set `QWEN_TRITON_STRICT=1` or pass `--strict-triton` to ensure no silent fallbacks to PyTorch. Any kernel failure will raise instead of falling back:

```bash
QWEN_TRITON_STRICT=1 CUDA_VISIBLE_DEVICES=3 python -m qwen_triton.scripts.smoke \
  --model-id Qwen/Qwen3-0.6B-Base \
  --backend triton --device cuda --dtype bf16 --compare-ref
```

## Test Commands

### CPU / Portable Unit Tests

```bash
pytest -q tests
```

### GPU Kernel Parity Tests

```bash
CUDA_VISIBLE_DEVICES=3 python -m pytest -q tests/test_kernels.py
```

### Benchmark Triton vs Reference on Wikitext

```bash
CUDA_VISIBLE_DEVICES=3 python -m qwen_triton.scripts.benchmark_wikitext \
  --model-id Qwen/Qwen3-0.6B-Base \
  --backends triton ref \
  --dataset wikitext-2-raw-v1 \
  --device cuda \
  --dtype bf16 \
  --batch-size 1 \
  --seq-len 128 \
  --train-steps 128 \
  --warmup-steps 2 \
  --eval-batches 8 \
  --lr 1e-5 \
  --output-dir artifacts/benchmarks/wikitext_compare
```

### Profile One Training Step with Nsight Systems

```bash
TMPDIR=/home/luo00466/Qwen-Triton/artifacts/profiles/tmp \
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=3 \
/usr/local/cuda/bin/nsys profile \
  --force-overwrite true \
  --stats=true \
  --sample=none \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  -o /home/luo00466/Qwen-Triton/artifacts/profiles/nsys_triton_train \
  /home/luo00466/.conda/envs/py310_2/bin/python -m qwen_triton.scripts.profile_backend_step \
    --model-id Qwen/Qwen3-0.6B-Base \
    --backend triton \
    --device cuda \
    --dtype bf16 \
    --mode train \
    --batch-size 1 \
    --seq-len 128 \
    --warmup-steps 1 \
    --profile-steps 1
```

Run the same command with `--backend ref` for the baseline.

## Measured Results On This Machine

Machine context:

- environment: `py310_2`
- model: `Qwen/Qwen3-0.6B-Base`
- dataset: `wikitext-2-raw-v1`
- run: batch size 1, sequence length 128, 128 train steps, 2 warmup steps, 8 eval batches, bf16

### End-to-End Fine-Tuning Comparison (Pre Full-Triton)

| Backend | Load Time (s) | Mean Train Step (ms) | Train Tok/s | Eval Loss | Eval Token Acc | Peak Mem (GB) | Total Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | 3.8667 | 77.78 | 1645.72 | 2.522461 | 0.505906 | 5.6037 | 19.3482 |
| Ref | 0.6762 | 77.59 | 1649.72 | 2.521912 | 0.506890 | 5.6048 | 11.1205 |

Note: these numbers were measured before the full-Triton conversion (when attention and GEMMs still used PyTorch/cuBLAS). The full-Triton path prioritizes completeness over performance -- the Triton GEMM kernel will be slower than cuBLAS for large shapes like the LM head (N=151936), but the entire model now runs through repo-owned Triton kernels.

## Artifacts

Benchmark artifacts:

- `artifacts/benchmarks/wikitext_compare_gpu5_seq128_128steps_post_gradfix/metrics.json`
- `artifacts/benchmarks/wikitext_compare_gpu5_seq128_128steps_opt2/metrics.json`
- `artifacts/benchmarks/wikitext_triton_only_seq128_128steps_opt2/metrics.json`

Profiler artifacts:

- `artifacts/profiles/nsys_triton_train_post_gradfix.nsys-rep`
- `artifacts/profiles/nsys_triton_train_post_gradfix.sqlite`
- `artifacts/profiles/nsys_triton_train_opt2.nsys-rep`
- `artifacts/profiles/nsys_triton_train_opt2.sqlite`
- `artifacts/profiles/nsys_ref_train_post_gradfix.nsys-rep`
- `artifacts/profiles/nsys_ref_train_post_gradfix.sqlite`

## Known Limitations

- Triton GEMM kernel is slower than cuBLAS for large matrix shapes (especially the LM head with vocab_size=151936). This is expected -- the goal was full Triton coverage, not peak GEMM throughput.
- Flash Attention uses fixed block sizes (BLOCK_M=32, BLOCK_N=32) to stay within shared memory limits. Larger block sizes would improve throughput on GPUs with more shared memory.
- Embedding backward uses atomic scatter-add, which may contend under large batch sizes with repeated tokens.
- Triton kernel autotune warmup adds significant startup cost compared to the reference backend.
- Qwen3.5 text-family and MoE paths are represented in code but not yet converted to full Triton.
- `ncu` currently prints a post-disconnect `utf-8-sig` traceback on this machine even when profiling succeeds and metrics are emitted.

## Next Performance Work

Now that the dense Qwen3 path is fully Triton, the highest-value next steps are:

1. **Tune Triton GEMM**: Add more autotune configs and split-K for shapes where Triton matmul significantly underperforms cuBLAS.
2. **Tune Flash Attention**: Increase block sizes on GPUs with larger shared memory; add sliding-window attention kernel variant.
3. **Reduce startup cost**: Cache compiled Triton kernels or precompile frequently used shapes.
4. **Extend to MoE**: Convert `QwenSparseMoeBlock` expert routing and expert MLPs to use `TritonLinear` and Triton softmax.
5. **Extend to Qwen3.5**: Tritonize the remaining linear-attention sub-ops and mixed-layer paths.
6. **Fused LM head + CE**: Combine the final linear projection with cross-entropy to avoid materializing the full vocab logits tensor.
7. **Decode optimization**: Add token-by-token generation microbenchmarks and optimize the cache-hit path.
