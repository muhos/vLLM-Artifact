#!/usr/bin/env python3
"""
Usage:
    python decode.py <model_path> [--eager] [--ncu] [--batch N]
                             [--gpu-mem FRAC] [--verbose]

    --eager      disable CUDA graphs
    --ncu        shrink the engine's memory budget to leave room for Nsight Compute
    --batch N    concurrent sequences, also tokens per decode step (default 64)
    --gpu-mem F  set the memory fraction explicitly, overriding the default
    --verbose    show vllm's own logging instead of capturing it
"""
import os
import sys
import re
import gc
import time
import logging
import contextlib

# Keeps vllm engine in this process (otherwise profilers see nothing).
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# ---------------------------------------------------------------------------
# Utilities for printing a simple report.
# ---------------------------------------------------------------------------
MAX_HEADER_LEN = 80

def make_header(title):
    rest_len = MAX_HEADER_LEN - (len(title) + 2) - 6
    print(f"\n{'-' * 6} [ {title} ] {'-' * rest_len}")


def row(label, value):
    print(f"{label:30}: {value}")

# ---------------------------------------------------------------------------
# Capture the engine measurements we care about out of vllm's log stream, then
# eat the line. Each entry is (keys, regex, type casts).
# ---------------------------------------------------------------------------
ENGINE = {}

def to_integer(s): return int(s.replace(",", ""))  # "10,000" -> 10000

CAPTURE = [
    (("weights_gib", "load_s"),         r"Model loading took ([\d.]+) GiB and ([\d.]+) seconds",            (float, float)),
    (("kv_gib",),                       r"Available KV cache memory: ([\d.]+) GiB",                         (float,)),
    (("kv_tokens",),                    r"GPU KV cache size: ([\d,]+) tokens",                              (to_integer,)),
    (("conc_ctx", "max_concurrency"),   r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x",  (to_integer, float)),
    (("graph_s", "graph_gib"),          r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB",       (float, float)),
    (("free_gib", "total_gib"),         r"Free memory on device \(([\d.]+)/([\d.]+) GiB\)",                 (float, float)),
    (("budget_gib",),                   r"Desired GPU memory utilization is \([\d.]+, ([\d.]+) GiB\)",      (float,)),
    (("activation_gib",),               r"([\d.]+) GiB for peak activation",                                (float,)),
    (("nontorch_gib",),                 r"([\d.]+) GiB for non-torch memory",                               (float,)),
    (("cudagraph_gib",),                r"([\d.]+) GiB for CUDAGraph memory",                               (float,)),
    (("attn_backend",),                 r"Using (\w+) attention backend",                                   (str,)),
    (("flash_version",),                r"Using FlashAttention version (\d+)",                              (str,)),
]

import torch
from vllm import LLM, SamplingParams

VERBOSE = "--verbose" in sys.argv

class CaptureEngineStats(logging.Filter):
    """Capture numbers out of vllm log lines, then drop the lines."""

    def filter(self, record):
        msg = record.getMessage()
        for keys, pattern, casts in CAPTURE:
            m = re.search(pattern, msg)
            if m:
                for k, raw, cast in zip(keys, m.groups(), casts):
                    ENGINE[k] = cast(raw)
        if record.levelno >= logging.WARNING:
            return True         # never hide a real warning or error
        return VERBOSE          # False: suppress the line entirely


capture_filter = CaptureEngineStats()
# Attach the filter to all vllm loggers, so we can capture the engine stats without printing them.
handlers = []
for logger_name in ("vllm", ""):  # "" is the root logger
    handlers.extend(logging.getLogger(logger_name).handlers)

for h in handlers:
    h.addFilter(capture_filter)

# ---------------------------------------------------------------------------
# Prepare the run parameters.
# ---------------------------------------------------------------------------
MODEL = sys.argv[1] if len(sys.argv) > 1 else sys.exit(
    "usage: decode.py <model_path> [--eager] [--ncu] [--batch N] [--gpu-mem FRAC]")
EAGER = "--eager" in sys.argv

# Default 0.9 matches sweep.sh, so KV cache and throughput numbers from the two
# scripts are directly comparable.
#
# --ncu gives Nsight Compute room for its kernel replay buffers, which need
# several GB of their own. Without it, ncu dies on allocation:
#   AWQ-INT4 8B  ~5.4 GB of weights: 0.5 leaves ~12 GB for ncu
#   FP16 8B     ~15.0 GB of weights: 0.85 leaves ~4.2 GB for ncu
quantized = any(t in MODEL.upper() for t in ("AWQ", "GPTQ", "INT4", "W4A16"))
GPU_MEM = 0.9
if "--ncu" in sys.argv:
    GPU_MEM = 0.5 if quantized else 0.85
if "--gpu-mem" in sys.argv:                 # explicit value beats both defaults
    GPU_MEM = float(sys.argv[sys.argv.index("--gpu-mem") + 1])

# Decode-heavy on purpose: short prompt, long generation.
PROMPT_TOKENS = 128
OUTPUT_TOKENS = 256
BATCH = 64
if "--batch" in sys.argv:
    BATCH = int(sys.argv[sys.argv.index("--batch") + 1])

# ---------------------------------------------------------------------------
# Engine start.
# ---------------------------------------------------------------------------
make_header("Engine")

llm_kwargs = dict(model=MODEL, max_model_len=2048, enforce_eager=EAGER, gpu_memory_utilization=GPU_MEM)

if not VERBOSE:
    llm_kwargs["use_tqdm_on_load"] = False
try:
    llm = LLM(**llm_kwargs)
except TypeError:
    llm_kwargs.pop("use_tqdm_on_load", None)
    llm = LLM(**llm_kwargs)

capture_dict = ENGINE.get

print("")

row("CUDA graphs", "disabled (eager)" if EAGER else "enabled")

attn_line = capture_dict("attn_backend", "?")
if capture_dict("flash_version"):
    attn_line += f" (FlashAttention v{capture_dict('flash_version')})"
row("Attention backend", attn_line)

if capture_dict("total_gib"):
    row("Device memory", f"{capture_dict('free_gib'):.2f} / {capture_dict('total_gib'):.2f} GB free at start")

budget_line = f"{GPU_MEM:.2f} of device"
if capture_dict("budget_gib"):
    budget_line += f"  ({capture_dict('budget_gib'):.2f} GB)"
row("Memory budget", budget_line)

if capture_dict("weights_gib"):
    row("Weights", f"{capture_dict('weights_gib'):.2f} GB  (loaded in {capture_dict('load_s'):.2f} sec)")

for label, key in (("Peak activation", "activation_gib"),
                   ("Non-torch memory", "nontorch_gib"),
                   ("CUDA graph memory", "cudagraph_gib")):
    if capture_dict(key) is not None:
        row(label, f"{capture_dict(key):.2f} GB")

if capture_dict("kv_gib") and capture_dict("kv_tokens"):
    row("KV cache", f"{capture_dict('kv_gib'):.2f} GB  ({capture_dict('kv_tokens'):,} tokens)")
    row("KV cache per token", f"{capture_dict('kv_gib') * 1024**3 / capture_dict('kv_tokens') / 1024:.0f} KB")

if capture_dict("max_concurrency"):
    row("Max concurrency", f"{capture_dict('max_concurrency'):.2f}x at {capture_dict('conc_ctx'):,} tokens/request")

if capture_dict("graph_gib"):
    row("Graph capture", f"{capture_dict('graph_s'):.0f} s, {capture_dict('graph_gib'):.2f} GB")

# How much cache is needed for this run, and how that compares to the engine's cache size.
needed = BATCH * (PROMPT_TOKENS + OUTPUT_TOKENS)
if capture_dict("kv_tokens"):
    percentage = needed / capture_dict("kv_tokens") * 100
    warn = "  <-- exceeds cache, expect queueing" if needed > capture_dict("kv_tokens") else ""
    row("This run needs", f"{needed:,} tokens ({percentage:.0f}% of cache){warn}")

# ---------------------------------------------------------------------------
# The actual run: warmup, then profiled decode.
# ---------------------------------------------------------------------------
prompt = "Let's profile this hardware to find interesting " + ("results " * PROMPT_TOKENS)
prompts = [prompt] * BATCH

# ignore_eos so every sequence generates exactly OUTPUT_TOKENS.
params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True)

make_header("Warmup (not profiled)")
llm.generate(prompts, params, use_tqdm=True)
torch.cuda.synchronize()

make_header("Generating sequences (profiled)")
torch.cuda.profiler.start()
time_start = time.perf_counter()

out = llm.generate(prompts, params, use_tqdm=True)
torch.cuda.synchronize()

elapsed = time.perf_counter() - time_start
torch.cuda.profiler.stop()

total_out = sum(len(o.outputs[0].token_ids) for o in out)

make_header("Results")

row("Model", MODEL)
row("Batch size", BATCH)
row("Prompt tokens per sequence", PROMPT_TOKENS)
row("Output tokens per sequence", OUTPUT_TOKENS)
row("Total output tokens", f"{total_out:,}")
row("Elapsed time", f"{elapsed:.3f} s")
row("Decode throughput", f"{total_out / elapsed:,.1f} tokens/s")
row("Per-step latency", f"{elapsed / OUTPUT_TOKENS * 1000:.2f} ms ({BATCH} tokens per step)")

# Weight traffic against the RTX 4090's 1008 GB/s, says how close decode contributes to the max memory bandwidth.
if capture_dict("weights_gib"):
    step_s = elapsed / OUTPUT_TOKENS
    weight_gb = capture_dict("weights_gib") * 1024**3 / 1e9
    gbs = weight_gb / step_s
    row("Weight traffic per step", f"{weight_gb:.2f} GB in {gbs:,.0f} GB/s ({gbs / 1008 * 100:.0f}% of peak)")

# ---------------------------------------------------------------------------
# Shut the engine down cleanly.
# ---------------------------------------------------------------------------
del llm, out
gc.collect()
with contextlib.suppress(Exception):
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    destroy_model_parallel()
    destroy_distributed_environment()
with contextlib.suppress(Exception):
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()