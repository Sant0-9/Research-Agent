#!/bin/bash
# vLLM Brain Service Startup Script
#
# This script starts the vLLM server with DeepSeek-R1-Distill-Qwen-14B
# Uses OpenAI-compatible API at /v1 endpoint

set -e

echo "Starting vLLM brain service..."
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "Max context length: ${MAX_MODEL_LEN}"

# Build vLLM command with all parameters
VLLM_ARGS=(
    "--model" "${MODEL_NAME}"
    "--host" "${HOST}"
    "--port" "${PORT}"
    "--max-model-len" "${MAX_MODEL_LEN}"
    "--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION}"
    "--dtype" "${DTYPE}"
    "--trust-remote-code"
    "--disable-log-requests"
)

# Add API key if provided
if [ -n "${BRAIN_API_KEY}" ]; then
    VLLM_ARGS+=("--api-key" "${BRAIN_API_KEY}")
    echo "API key authentication enabled"
fi

# Add tensor parallel if multiple GPUs
if [ -n "${TENSOR_PARALLEL_SIZE}" ] && [ "${TENSOR_PARALLEL_SIZE}" -gt 1 ]; then
    VLLM_ARGS+=("--tensor-parallel-size" "${TENSOR_PARALLEL_SIZE}")
    echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
fi

# Add quantization if specified
if [ -n "${QUANTIZATION}" ]; then
    VLLM_ARGS+=("--quantization" "${QUANTIZATION}")
    echo "Quantization: ${QUANTIZATION}"
fi

# Enable chunked prefill for long contexts
if [ "${ENABLE_CHUNKED_PREFILL}" = "true" ]; then
    VLLM_ARGS+=("--enable-chunked-prefill")
    echo "Chunked prefill enabled"
fi

# Launch vLLM
echo "Starting vLLM with args: ${VLLM_ARGS[*]}"
exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
