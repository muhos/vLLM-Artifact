#!/usr/bin/env python3
"""
    python analyze.py [results_dir]        # default ./results

Outputs into the results dir:
    summary.csv            one row per run
    throughput.png         fp16 vs awq at peak load
    latency_vs_rate.png    TTFT / TPOT vs request rate
    kvcache.png            KV cache capacity per model
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")           # headless box
import matplotlib.pyplot as plt 

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "./results")

if not RESULTS.is_dir():
    sys.exit(f"{RESULTS} is not a directory or does not exist")

# Only the metrics we actually report.
FIELDS = [
    "duration", "completed", "request_rate",
    "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput", "total_token_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
]

def load_runs(d: Path) -> pd.DataFrame:
    rows = []
    for jf in sorted(d.glob("bench_*.json")):
        m = re.match(r"bench_(\w+?)_(.+)", jf.stem)
        if not m:
            continue
        try:
            data = json.loads(jf.read_text())
        except Exception as e:
            print(f"Error: {jf.name}: {e}")
            continue
        row = {"model": m.group(1), "config": m.group(2)}
        row.update({f: data.get(f) for f in FIELDS})
        if row.get("completed"):
            row["input_len"] = round(row["total_input_tokens"] / row["completed"])
            row["output_len"] = round(row["total_output_tokens"] / row["completed"])
        rows.append(row)
    if not rows:
        sys.exit(f"no bench_*.json in {d}")
    return pd.DataFrame(rows).sort_values(["model", "config"])

def load_kv(d: Path) -> dict:
    """KV cache token capacity per model, from the lines sweep.sh saved."""
    out = {}
    for kf in d.glob("kv_*.txt"):
        for line in kf.read_text().splitlines():
            if "GPU KV cache size" in line:
                digits = line.split(":")[-1].replace("tokens", "").strip()
                try:
                    out[kf.stem.replace("kv_", "")] = int(digits.replace(",", ""))
                except ValueError:
                    pass
    return out

def plot_throughput(df, path):
    peak = df[df["config"].str.endswith("_peak")]
    if peak.empty:
        return
    ax = peak.pivot_table(index="config", columns="model", values="output_throughput") \
             .plot(kind="bar", figsize=(9, 5), rot=15)
    ax.set_ylabel("Output token throughput (tok/s)")
    ax.set_xlabel("")
    ax.set_title("Peak decode throughput in Llama-3.1-8B, RTX 4090")
    ax.grid(axis="y", alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  generated {path.name}")


def plot_latency(df, path):
    rate_runs = df[df["config"].str.startswith("rps")].copy()
    if rate_runs.empty:
        return
    rate_runs["rate"] = pd.to_numeric(rate_runs["request_rate"], errors="coerce")
    rate_runs = rate_runs.dropna(subset=["rate"]).sort_values("rate")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, g in rate_runs.groupby("model"):
        axes[0].plot(g["rate"], g["median_ttft_ms"], marker="o", label=model)
        axes[1].plot(g["rate"], g["mean_tpot_ms"], marker="o", label=model)
    axes[0].set(title="Time to first token (prefill + queue)", ylabel="Median TTFT (ms)")
    axes[1].set(title="Time per output token (decode)", ylabel="Mean TPOT (ms)")
    for a in axes:
        a.set_xlabel("Request rate (req/s)")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("Latency vs request rate (1024 in / 128 out)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  generated {path.name}")


def plot_kv(kv, path):
    if not kv:
        return
    fig, ax = plt.subplots(figsize=(5, 4.5))
    keys = list(kv)
    ax.bar(keys, [kv[k] for k in keys], color=["#4472c4", "#ed7d31"][:len(keys)])
    ax.set_ylabel("KV cache capacity (tokens)")
    ax.set_title("KV cache available for use after weights")
    ax.grid(axis="y", alpha=0.3)
    for i, k in enumerate(keys):
        ax.text(i, kv[k], f"{kv[k]:,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  generated {path.name}")


def main():
    df = load_runs(RESULTS)
    df.to_csv(RESULTS / "summary.csv", index=False)

    cols = ["model", "config", "input_len", "output_len", "output_throughput",
            "median_ttft_ms", "mean_tpot_ms", "median_itl_ms"]
    with pd.option_context("display.width", 160):
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    kv = load_kv(RESULTS)
    if kv:
        print("\nKV cache capacity (tokens):")
        for k, v in kv.items():
            print(f"  {k:>5}: {v:,}")
        if {"fp16", "awq"} <= set(kv):
            print(f"  awq gives {kv['awq'] / kv['fp16']:.2f}x of availability vs fp16")

    print()
    plot_throughput(df, RESULTS / "throughput.png")
    plot_latency(df, RESULTS / "latency_vs_rate.png")
    plot_kv(kv, RESULTS / "kvcache.png")

    p = df.pivot_table(index="config", columns="model", values="output_throughput")
    if {"fp16", "awq"} <= set(p.columns):
        p["speedup"] = p["awq"] / p["fp16"]
        print("\nOutput throughput speedup:\n")
        print(p[["fp16", "awq", "speedup"]].reset_index().to_string(
            index=False,
            formatters={"fp16": "{:,.0f}".format, "awq": "{:,.0f}".format, "speedup": "{:.2f}x".format}))


if __name__ == "__main__":
    main()
