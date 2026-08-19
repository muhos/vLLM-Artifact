#!/bin/bash

for M in meta-llama_Llama-3.1-8B-Instruct hugging-quants_Meta-Llama-3.1-8B-Instruct-AWQ-INT4
do
    for MODE in graphs eager
        do FLAG=""
        [ "$MODE" = eager ] && FLAG="--eager"
        printf "===============[ %-50s (%-5s) ]================\n" "$M" "$MODE"
        python decode.py models/$M $FLAG --batch 64 --gpu-mem 0.9 2>&1 | tail -22
    done
done 2>&1 | tee logs/eager_vs_graphs.log