#!/usr/bin/env bash

# 1. Move to workspace root
cd /scratch/hpc-prf-dssecs/pg_arc_ref/hpc_api || exit 1

# 2. Load necessary Apptainer module
echo "Loading Apptainer module..."
module load tools/Apptainer

# 3. Model path configuration (Default: Qwen3-Coder, TP=2)
MODEL_PATH="${1:-./models/Qwen3-Coder-30B-A3B-Instruct}"
TP_SIZE="${2:-2}"

echo "=================================================="
echo "Starting vLLM Server"
echo "Model: ${MODEL_PATH}"
echo "Tensor Parallel Size: ${TP_SIZE}"
echo "=================================================="

# 4. Launch vLLM container
PYTHONNOUSERSITE=1 apptainer run --nv ./vllm_qwen3.sif \
    "${MODEL_PATH}" \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --max-model-len 16384 \
    --tensor-parallel-size "${TP_SIZE}" \
    --distributed-executor-backend mp

