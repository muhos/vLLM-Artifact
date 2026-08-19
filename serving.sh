#!/usr/bin/env bash
# =============================================================================
# Starts a vLLM server per model, runs every query against it, shuts it down.
# One result JSON per run.
#
#   ./sweep.sh
#   python analyze.py
# =============================================================================
set -uo pipefail

LAB=$PWD
RESULTS=$LAB/results
LOGS=$LAB/logs
PORT=8000
mkdir -p "$RESULTS" "$LOGS"

MODELS=(
    "fp16:$LAB/models/meta-llama_Llama-3.1-8B-Instruct"
    "awq:$LAB/models/hugging-quants_Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
)

# name | input_len | output_len | request_rate | num_prompts
#   *_peak : every request sent at once.
#   rpsN   : N requests per second arriving.
CONFIGS=(
    "prefill_peak|1024|128|inf|200"
    "balanced_peak|512|512|inf|150"
    "decode_peak|256|1024|inf|100"
    "rps4|1024|128|4|80"
    "rps8|1024|128|8|120"
    "rps16|1024|128|16|160"
)

# Poll /health until the server answers, or the process dies.
wait_ready() {
    for _ in $(seq 100); do
        curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "$1" 2>/dev/null || return 1
        sleep 3
    done
    return 1
}

server_pid=""
stop_server() {
    [[ -z "$server_pid" ]] && return
    kill -INT "$server_pid" 2>/dev/null
    wait "$server_pid" 2>/dev/null
    server_pid=""
    sleep 5  # let the GPU memory actually free
}
trap 'echo; stop_server; exit 130' INT TERM

for entry in "${MODELS[@]}"; do
    KEY=${entry%%:*}
    MPATH=${entry#*:}
    [[ -f "$MPATH/config.json" ]] || { echo "warning: no config.json for $KEY, skipping"; continue; }

    echo
    printf "===============[ %-5s ]================\n" "$KEY"

    vllm serve "$MPATH" --max-model-len 8192 --port "$PORT" > "$LOGS/server_$KEY.log" 2>&1 &

    server_pid=$!

    if ! wait_ready "$server_pid"; then
        echo "error: server never launched; see $LOGS/server_$KEY.log"
        stop_server
        continue
    fi

    grep -oE "(Available KV cache memory|GPU KV cache size|Maximum concurrency).*" "$LOGS/server_$KEY.log" | tail -3 | tee "$RESULTS/kv_$KEY.txt"

    for cfg in "${CONFIGS[@]}"; do

        IFS='|' read -r NAME ILEN OLEN RATE N <<< "$cfg"
        OUT="bench_${KEY}_${NAME}.json"

        [[ -f "$RESULTS/$OUT" ]] && { echo " $KEY/$NAME already done"; continue; }

        echo " $KEY/$NAME  (in=$ILEN out=$OLEN rate=$RATE n=$N)"
        vllm bench serve --backend vllm --model "$MPATH" \
            --dataset-name random \
            --random-input-len "$ILEN" --random-output-len "$OLEN" \
            --num-prompts "$N" --request-rate "$RATE" \
            --save-result --result-dir "$RESULTS" --result-filename "$OUT" \
            2>&1 | tee "$LOGS/bench_${KEY}_${NAME}.log" \
            | grep -E "Output token throughput|Median TTFT|Mean TPOT|Successful requests" | sed 's/^/   /'
    done

    stop_server
done

echo
echo "done"

python analyze.py
