# Qwen-Triton

Qwen-Triton is a repo-owned Qwen3/Qwen3.5 bring-up that keeps the public model interface stable while progressively swapping PyTorch/Hugging Face pieces for Triton and custom CUDA implementations.

The current validated target is text-only `Qwen/Qwen3-0.6B-Base`. Dense Qwen3 checkpoint load, forward, generation, and short Wikitext fine-tuning all run on GPU. Qwen3 MoE and Qwen3.5 text-family support are scaffolded in the config and module layers, but they have not yet been profiled and benchmarked as thoroughly as dense Qwen3.

Important: `backend="triton"` currently means "repo-owned hybrid backend", not "every heavy op is implemented in Triton". Attention uses Torch SDPA / CUDA attention kernels, GEMMs still go through PyTorch/cuBLAS, and only part of the training path is fully Tritonized today.

## Current Status

- Validated model target: `Qwen/Qwen3-0.6B-Base`
- Validated environment: `py310_2`
- Validated debug GPU: `CUDA_VISIBLE_DEVICES=5`
- Working backends:
  - `ref`: upstream Hugging Face model for parity and baseline measurement
  - `triton`: repo-owned module stack with Triton kernels and optional CUDA RoPE op
- Working RoPE backends:
  - default Triton RoPE path
  - explicit CUDA custom operator via `QWEN_TRITON_ROPE_BACKEND=cuda_op`
- Fine-tuning correctness bug fixed:
  - the original Triton RMSNorm / SiLU-mul / RoPE path was forward-only
  - this broke autograd and caused long-run training drift
  - Triton backward kernels now exist for RMSNorm, SiLU-mul, and RoPE
  - training now matches the reference path again
- Decode cache path improved:
  - full-attention KV cache no longer has to grow by append-and-copy on every decode step
  - cache storage is now reused with write-by-position updates and capacity growth
  - this keeps the repo-owned generation path closer to a real serving cache design
- Checkpoint load path improved:
  - repo load no longer random-initializes the whole model before loading checkpoint tensors
  - this reduced Triton load time materially on the tested Qwen3-0.6B path

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

- token embeddings
- a decoder stack of `QwenDecoderLayer`
- final RMSNorm
- tied LM head when configured

Each decoder layer does:

1. input RMSNorm
2. attention or linear-attention block
3. residual add
4. post-attention RMSNorm
5. dense MLP or sparse MoE block
6. residual add

Dense Qwen3 is the most validated path today.

### 3. Attention

`QwenFullAttention` is repo-owned, but not yet full-Triton:

- Q/K/V/O projections are standard PyTorch `nn.Linear`
- Q/K head normalization is repo-owned RMSNorm
- rotary embedding is repo-owned and can run with Triton or a CUDA custom op
- KV cache update is repo-owned and now uses preallocated write-by-position storage
- attention score/value computation uses `torch.nn.functional.scaled_dot_product_attention`

That design keeps the architecture under repo control while deferring the biggest kernel work until the surrounding model behavior is stable.

### 4. MLP / MoE / Qwen3.5

Dense MLP uses the standard Qwen SwiGLU structure:

- `gate_proj`
- `up_proj`
- fused SiLU-mul epilogue
- `down_proj`

MoE and Qwen3.5 linear-attention blocks are scaffolded so the repo can represent:

- dense Qwen3
- sparse Qwen3 MoE
- Qwen3.5 text blocks with mixed layer types
- shared-expert style Qwen3.5 text-family MoE

The dense Qwen3 path is the only path fully exercised end-to-end on a real checkpoint in this README.

### 5. Triton and CUDA Kernels

Current repo-owned kernel layer:

- `qwen_triton/kernels/rmsnorm.py`
- `qwen_triton/kernels/swiglu.py`
- `qwen_triton/kernels/sigmoid_mul.py`
- `qwen_triton/kernels/rope.py`
- `qwen_triton/kernels/cache.py`
- `qwen_triton/kernels/linear_attention.py`

Current explicit CUDA custom operator:

- `qwen_triton/csrc/rope_op.cpp`
- `qwen_triton/csrc/rope_op_kernel.cu`
- Python wrapper in `qwen_triton/ops/rope_cuda.py`

The RoPE path is selectable by environment variable:

```bash
export QWEN_TRITON_ROPE_BACKEND=triton
export QWEN_TRITON_ROPE_BACKEND=cuda_op
export QWEN_TRITON_ROPE_BACKEND=auto
```

### 6. Loader Design

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
  - decoder blocks, attention, rotary, MLP, MoE, cache
- `qwen_triton/kernels`
  - Triton kernels and their runtime wrappers
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

The biggest training bug found during validation was that the original Triton primitives for:

- RMSNorm
- SiLU-mul
- RoPE

were forward-only and therefore detached from autograd. The result was acceptable short forward parity but incorrect long-run fine-tuning behavior.

That issue is now fixed by restoring autograd through the Triton primitive wrappers. In practice that means:

- long-run Wikitext validation matches the reference model again
- kernel-level forward and gradient parity match the torch reference on CUDA tensors
- training is correct
- RMSNorm and SiLU-mul now have Triton backward kernels
- attention output gating now has a fused Triton `sigmoid_mul` kernel for gated-attention families
- the backend is much closer to the reference training step speed, but still not clearly faster end-to-end

This is the current honest state of the repo: correctness first, then performance tuning.

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

The validated local debug environment is still `py310_2`, but `environment.yml` and `requirements.txt` are now the repo-supported setup path for new users.

Build the CUDA RoPE operator after the environment is active with:

```bash
TORCH_CUDA_ARCH_LIST=12.0 python -m qwen_triton.scripts.build_rope_cuda_op --verbose
```

## Quick Start

### Smoke Test

```bash
CUDA_VISIBLE_DEVICES=5 python -m qwen_triton.scripts.smoke \
  --model-id Qwen/Qwen3-0.6B-Base \
  --backend triton \
  --device cuda \
  --dtype bf16 \
  --max-new-tokens 4 \
  --compare-ref
```

### One-Step Wikitext Train/Eval Smoke

```bash
CUDA_VISIBLE_DEVICES=5 python -m qwen_triton.scripts.train_wikitext \
  --model-id Qwen/Qwen3-0.6B-Base \
  --backend triton \
  --dataset wikitext-2-raw-v1 \
  --device cuda \
  --dtype bf16 \
  --seq-len 32 \
  --train-steps 1 \
  --eval-batches 1
```

## Test Commands

### CPU / Portable Unit Tests

```bash
pytest -q tests
```

### GPU Kernel Parity Tests

```bash
CUDA_VISIBLE_DEVICES=5 python -m pytest -q tests/test_kernels.py
```

### Benchmark Triton vs Reference on Wikitext

```bash
CUDA_VISIBLE_DEVICES=5 python -m qwen_triton.scripts.benchmark_wikitext \
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
  --output-dir artifacts/benchmarks/wikitext_compare_gpu5_seq128_128steps_post_gradfix
```

### Profile One Training Step with Nsight Systems

```bash
TMPDIR=/home/luo00466/Qwen-Triton/artifacts/profiles/tmp \
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=5 \
/usr/local/cuda/bin/nsys profile \
  --force-overwrite true \
  --stats=true \
  --sample=none \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  -o /home/luo00466/Qwen-Triton/artifacts/profiles/nsys_triton_train_post_gradfix \
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

Run the same command with `--backend ref` and output path `nsys_ref_train_post_gradfix` for the baseline.

### Profile Specific Kernels with Nsight Compute

Triton RMSNorm:

```bash
TMPDIR=/home/luo00466/Qwen-Triton/artifacts/profiles/tmp \
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=5 \
/usr/local/cuda/bin/ncu \
  --target-processes all \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
  --kernel-name-base demangled \
  -k regex:_rmsnorm_kernel \
  --launch-count 1 \
  --nvtx \
  --nvtx-include 'triton_train_profile]' \
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

Triton SiLU-mul:

```bash
TMPDIR=/home/luo00466/Qwen-Triton/artifacts/profiles/tmp \
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=5 \
/usr/local/cuda/bin/ncu \
  --target-processes all \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
  --kernel-name-base demangled \
  -k regex:_silu_mul_kernel \
  --launch-count 1 \
  --nvtx \
  --nvtx-include 'triton_train_profile]' \
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

Reference flash-attention forward:

```bash
TMPDIR=/home/luo00466/Qwen-Triton/artifacts/profiles/tmp \
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=5 \
/usr/local/cuda/bin/ncu \
  --target-processes all \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
  --kernel-name-base demangled \
  -k regex:flash_fwd_kernel \
  --launch-count 1 \
  --nvtx \
  --nvtx-include 'ref_train_profile]' \
  /home/luo00466/.conda/envs/py310_2/bin/python -m qwen_triton.scripts.profile_backend_step \
    --model-id Qwen/Qwen3-0.6B-Base \
    --backend ref \
    --device cuda \
    --dtype bf16 \
    --mode train \
    --batch-size 1 \
    --seq-len 128 \
    --warmup-steps 1 \
    --profile-steps 1
```

## Measured Results On This Machine

Machine context:

- environment: `py310_2`
- GPU: `CUDA_VISIBLE_DEVICES=5`
- model: `Qwen/Qwen3-0.6B-Base`
- dataset: `wikitext-2-raw-v1`
- run: batch size 1, sequence length 128, 128 train steps, 2 warmup steps, 8 eval batches, bf16

### End-to-End Fine-Tuning Comparison

| Backend | Load Time (s) | Mean Train Step (ms) | Train Tok/s | Eval Loss | Eval Token Acc | Peak Mem (GB) | Total Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | 3.8667 | 77.78 | 1645.72 | 2.522461 | 0.505906 | 5.6037 | 19.3482 |
| Ref | 0.6762 | 77.59 | 1649.72 | 2.521912 | 0.506890 | 5.6048 | 11.1205 |

Interpretation:

- Training correctness now matches the reference path.
- After adding Triton backward kernels for RMSNorm and SiLU-mul plus a fused `sigmoid_mul` op, the Triton backend is now essentially tied with the reference backend on mean train-step time for this workload.
- The remaining performance gap is mostly not RoPE itself; it is a combination of:
  - slower startup/warmup
  - more generic elementwise work in the corrected training path
  - the fact that RoPE backward is still not a real Triton kernel

- Compared with the earlier corrected Triton path, the optimized Triton kernels reduced mean train-step time from `90.26 ms` to `77.78 ms` on the same seq128 / 128-step benchmark.
- End-to-end runtime is still worse because Triton startup, kernel warmup, and load-time overhead remain much larger than the reference path.

### Nsight Systems Summary

NVTX step ranges:

- Triton warmup: `5184.449 ms`
- Triton profiled train step: `103.581 ms`
- Ref warmup: `675.530 ms`
- Ref profiled train step: `110.905 ms`

Top Triton trace kernels by aggregate time:

- `vectorized_elementwise_kernel`: `3262.036 ms`
- `_rmsnorm_backward_kernel`: `84.941 ms`
- `_rmsnorm_kernel`: `54.004 ms`
- `multi_tensor_apply_kernel`: `33.008 ms`
- `_silu_mul_backward_kernel`: `26.349 ms`
- `_silu_mul_kernel`: `22.611 ms`
- `Kernel2`: `14.209 ms`
- `fmha_cutlassB_f32_aligned_64x64_k128_sm80`: `3.320 ms`
- `_rope_tensor_kernel`: `0.187 ms`

Top reference trace kernels by aggregate time:

- `multi_tensor_apply_kernel`: `33.062 ms`
- `Kernel2`: `14.168 ms`
- `vectorized_elementwise_kernel`: `7.324 ms`
- `elementwise_kernel`: `5.508 ms`
- `reduce_kernel`: `2.726 ms`
- `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`: `0.505 ms`
- `flash_fwd_kernel`: `0.486 ms`

What this says:

- the optimized Triton path now clearly routes RMSNorm and SiLU backward through custom Triton kernels
- the profiled Triton train step is now faster than the last reference profile step on this workload
- the remaining end-to-end problem is dominated by startup and autotune warmup, not by the steady-state train step
- generic elementwise work is still large across the full trace, especially during warmup
- RoPE cost is tiny in the full training trace

### Nsight Compute Highlights

Representative utilization samples:

- Triton `_rmsnorm_kernel`
  - DRAM throughput: `1.61%`
  - SM throughput: `3.35%`
  - active warps: `8.20%`
- Triton `_silu_mul_kernel`
  - DRAM throughput: `25.35%`
  - SM throughput: `8.66%`
  - active warps: `55.97%`
- Reference `flash_fwd_kernel`
  - DRAM throughput: `5.01%`
  - SM throughput: `2.68%`
  - active warps: `8.29%`

Interpretation:

- `_rmsnorm_kernel` is clearly underutilizing the GPU on this workload
- `_silu_mul_kernel` has much healthier occupancy, but it is not large enough to dominate total runtime
- the step-level slowdown is therefore more about the broader execution mix than a single catastrophic kernel

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

- The repo-owned backend is still hybrid, not full-Triton.
- Attention and GEMM kernels are not yet repo-owned Triton kernels.
- Training correctness is fixed, and train-step speed is now much closer to the reference baseline, but end-to-end runtime is still not better.
- Qwen3.5 text-family and MoE paths are represented in code but not yet validated as deeply as dense Qwen3.
- `ncu` currently prints a post-disconnect `utf-8-sig` traceback on this machine even when profiling succeeds and metrics are emitted.

## Next Performance Work

If the goal is to move from "correct and runnable" to "actually faster than baseline", the highest-value next steps are:

1. Reduce Triton startup/warmup cost by caching or precompiling frequently used kernel shapes.
2. Move the attention path away from Torch SDPA toward a repo-owned Triton attention implementation.
3. Tritonize more Qwen3.5 linear-attention sub-ops so that mixed-layer Qwen3.5 paths stop leaning on generic torch elementwise work.
4. Add decode-time microbenchmarks and profiling focused on cache reuse and token-by-token generation throughput.
5. Keep load-time work off the critical path by further optimizing weight mapping and device transfer.
