#!/usr/bin/env bash
# =============================================================================
# Per-kernel roofline data from Nsight Compute: where decode time goes, and
# whether each kernel is limited by memory bandwidth or by compute.
#
# Benchmarks: {fp16, awq} x {batch 16, 64}.
#
# All runs use --eager. CUDA graphs hide per-kernel attribution, and they are
# only worth 3-4% of throughput anyway (measured with decode.py).
#
#   ./roofline.sh
#   python analyze_roofline.py
# =============================================================================
set -uo pipefail

LAB=$PWD
RESULTS=$LAB/results
LOGS=$LAB/logs
mkdir -p "$RESULTS" "$LOGS"

MODELS=(
    "fp16:$LAB/models/meta-llama_Llama-3.1-8B-Instruct"
    "awq:$LAB/models/hugging-quants_Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
)
BATCHES=(16 64)

METRICS="dram__bytes_read.sum,\
dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
gpu__time_duration.sum"

for entry in "${MODELS[@]}"; do
    KEY=${entry%%:*}
    MPATH=${entry#*:}
    [[ -f "$MPATH/config.json" ]] || { echo "warning: no config.json for $KEY, skipping"; continue; }

    for B in "${BATCHES[@]}"; do
        TAG="${KEY}_b${B}"
        OUT="$RESULTS/ncu_${TAG}.csv"
        [[ -f "$OUT" ]] && { echo "-- $TAG already done"; continue; }

        echo
        printf "===============[ %-10s ]================\n" "$TAG"

        # real numbers first before profiling overhead.
        echo " run (no profiler)"
        python decode.py "$MPATH" --eager --batch "$B" \
            > "$LOGS/base_${TAG}.log" 2>&1
        grep -E "Decode throughput|Per-step latency|Total DRAM traffic" \
            "$LOGS/base_${TAG}.log" | sed 's/^/    /' \
            || echo "    failed, see $LOGS/base_${TAG}.log"

        # By default ncu replays each kernel in place, which means saving and
        # restoring every byte that kernel writes. FP16 weights already take
        # 15 of the 24 GiB, so there is no room for those buffers and ncu dies
        # mid-run ("Failed to profile act_and_mul_kernel", error code 9).
        # Application replay re-runs the whole program once per pass instead,
        # which needs no replay buffers. It is slower, and only valid because
        # this workload is deterministic (temperature 0, ignore_eos, fixed
        # prompts), so every run issues the same kernels with the same shapes.
        NCU_REPLAY=""
        [[ "$KEY" == "fp16" ]] && NCU_REPLAY="--replay-mode application"

        echo " ncu $NCU_REPLAY"
        ncu --profile-from-start off \
            --clock-control none \
            $NCU_REPLAY \
            --launch-skip 200 \
            --launch-count 60 \
            --metrics "$METRICS" \
            --csv --log-file "$OUT" \
            python decode.py "$MPATH" --eager --ncu --batch "$B" \
            > "$LOGS/ncu_${TAG}.log" 2>&1

        if [[ -s "$OUT" ]]; then
            echo "    wrote $(basename "$OUT") ($(wc -l < "$OUT") lines)"
        else
            echo "    no output, see $LOGS/ncu_${TAG}.log"
            rm -f "$OUT"        # an empty file would block a retry
        fi
        sleep 5
    done
done

echo
echo "done"

python analyze_roofline.py