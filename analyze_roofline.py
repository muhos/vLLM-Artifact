#!/usr/bin/env python3
"""
analyze_roofline.py — compare the ncu_*.csv files roofline.sh produced.

    python analyze_roofline.py [results_dir]     # default ~/llm-lab/results

Per run: which kernels own the time, and whether each is near the memory
ceiling or the compute ceiling. Then one table comparing the dominant GEMM
across models and batch sizes -- launch geometry and utilisation, which is
what survives the profiler intact.

Wall-clock is NOT reported: ncu replays kernels, so its timings run several
times slow. Take throughput and bandwidth from decode.py instead.
"""
import re
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "llm-lab/results")
METRIC = {
    "read": "dram__bytes_read.sum",
    "write": "dram__bytes_write.sum",
    "dram": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "warp": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "time": "gpu__time_duration.sum",
}


def kernel_group(name: str) -> str:
    """Collapse mangled kernel names into the handful of roles that matter.

    The order of these tests matters, because the mangled names overlap:
      - flash_fwd_splitkv_kernel carries cutlass template types, so it must be
        matched before the GEMM patterns or attention lands in the GEMM bucket.
      - reshape_and_cache_flash_kernel contains "flash", so it must be matched
        before the attention patterns.
    """
    n = str(name).lower()
    if "marlin" in n:
        return "GEMM (marlin int4)"
    if "reshape_and_cache" in n:
        return "kv_cache_write"
    if any(t in n for t in ("flash_fwd", "paged_attention", "attention")):
        return "attention"
    if any(t in n for t in ("cutlass", "gemm", "s16816", "gemv")):
        return "GEMM (fp16)"
    if "act_and_mul" in n:
        return "act_and_mul"
    if "rotary" in n:
        return "rotary"
    if "norm" in n:
        return "norm"
    m = re.match(r"\s*(?:void\s+)?([a-z_][\w:]*)", n)
    return (m.group(1) if m else n[:24]).split("<")[0][:22]


def load(path: Path) -> pd.DataFrame:
    with path.open() as f:
        skip = next(i for i, line in enumerate(f) if line.startswith('"ID"'))
    raw = pd.read_csv(path, skiprows=skip)
    raw["Metric Value"] = pd.to_numeric(
        raw["Metric Value"].astype(str).str.replace(",", ""), errors="coerce")
    raw["k"] = raw["Kernel Name"].map(kernel_group)

    wide = raw.pivot_table(index=["ID", "k"], columns="Metric Name",
                           values="Metric Value", aggfunc="first").reset_index()
    have = {k: v for k, v in METRIC.items() if v in wide.columns}

    agg = wide.groupby("k").agg(
        launches=("ID", "count"),
        us=(have["time"], lambda s: s.sum() / 1000),
        read_MB=(have["read"], lambda s: s.sum() / 1e6),
        dram_pct=(have["dram"], "mean"),
        sm_pct=(have["sm"], "mean"),
    )
    if "warp" in have:
        agg["warp_pct"] = wide.groupby("k")[have["warp"]].mean()
    grid = raw.groupby("k")["Grid Size"].agg(
        lambda s: s.mode().iat[0] if len(s) else "")
    return agg.join(grid).sort_values("us", ascending=False)


def main():
    files = sorted(RESULTS.glob("ncu_*.csv"))
    if not files:
        sys.exit(f"no ncu_*.csv in {RESULTS}")

    rows = []
    for f in files:
        m = re.match(r"ncu_(fp16|awq)_b(\d+)", f.stem)
        if not m:
            print(f"  ! skipping {f.name} (unexpected name)")
            continue
        model, batch = m.group(1), int(m.group(2))
        try:
            g = load(f)
        except Exception as e:                                  # noqa: BLE001
            print(f"  ! {f.name}: {e}")
            continue

        print(f"\n{'=' * 74}\n{model} / batch {batch}   ({f.name})\n{'=' * 74}")
        print(g.to_string(float_format=lambda x: f"{x:,.2f}"))

        gemm = g[g.index.str.startswith("GEMM")]
        if gemm.empty:
            continue
        top = gemm.iloc[0]
        blocks = 1
        for part in re.findall(r"\d+", str(top["Grid Size"])):
            blocks *= int(part)

        rows.append({
            "model": model, "batch": batch,
            "gemm_time_pct": gemm["us"].sum() / g["us"].sum() * 100,
            "gemm_grid": str(top["Grid Size"]),
            "blocks": blocks,
            "gemm_dram_pct": top["dram_pct"],
            "gemm_sm_pct": top["sm_pct"],
            "gemm_warp_pct": top.get("warp_pct", float("nan")),
        })

    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["model", "batch"])

    # Deliberately no step time, throughput or GB/s here. ncu replays kernels,
    # so its wall-clock runs several times slow -- and by different amounts per
    # replay mode, which makes even the ratios between runs wrong. Those numbers
    # come from decode.py with no profiler attached. What survives the profiler
    # is what this table shows: launch geometry, utilisation, and time shares
    # within a single run.
    print(f"\n{'#' * 74}\nGEMM per model and batch\n{'#' * 74}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    for model, g in df.groupby("model"):
        if {16, 64} <= set(g["batch"]):
            a = g[g["batch"] == 16].iloc[0]
            z = g[g["batch"] == 64].iloc[0]
            verdict = "grows with batch" if z.blocks > a.blocks else "FIXED"
            print(f"\n  {model}: {a.blocks} -> {z.blocks} blocks ({verdict}), "
                  f"dram {a.gemm_dram_pct:.0f}% -> {z.gemm_dram_pct:.0f}%")


if __name__ == "__main__":
    main()
