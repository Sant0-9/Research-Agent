# vLLM Brain Service Dockerfile
# Runs DeepSeek-R1-Distill-Qwen-14B with OpenAI-compatible API
#
# Build: docker build -t research-agent-brain:latest .
# Run: docker run --gpus all -p 8001:8001 research-agent-brain:latest

FROM vllm/vllm-openai:v0.13.0

# Set environment variables
ENV MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
ENV PORT=8001
ENV HOST="0.0.0.0"
ENV MAX_MODEL_LEN=131072
ENV GPU_MEMORY_UTILIZATION=0.90
ENV DTYPE="auto"

# Create non-root user for security
RUN useradd -m -u 1000 vllm && \
    mkdir -p /app /models && \
    chown -R vllm:vllm /app /models

# Set working directory
WORKDIR /app

# Copy startup script
COPY --chown=vllm:vllm start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Switch to non-root user
USER vllm

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command
CMD ["/app/start.sh"]
