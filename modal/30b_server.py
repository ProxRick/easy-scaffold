# Nomos-1 vLLM Server Configuration
# Optimized for 32B parameter MoE model (Qwen3-30B-A3B-Thinking-2507 based)
# Model card: https://huggingface.co/NousResearch/nomos-1

import asyncio
import json
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

import aiohttp
import modal

# -----------------------------
# Model & server configuration
# -----------------------------

# Nomos-1: 32B parameter MoE model specialized for mathematical problem-solving
# Model card: https://huggingface.co/NousResearch/nomos-1
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507"
MODEL_REVISION = None  # Use latest revision

# Hardware: Nomos-1 (32B MoE) is much smaller than DeepSeek-Math-V2 (685B)
# Memory calculation: 32B params = ~64GB total, with TP=4 = ~16GB/GPU + KV cache (~15GB) = ~31GB/GPU
# A100-80GB or H100-80GB work perfectly with TP=4 (plenty of headroom)
# H200-141GB is overkill but works great
# For "doodling": TP=4 is perfect, TP=2 also works but tighter
N_GPU = 4  # 4× A100-80GB or H100-80GB GPUs (cost-effective for experimentation)

# Tensor parallelism: Model card recommends TP=8 for optimal performance
# For cost-effective deployment: TP=4 works great (uses ~31-36GB per GPU)
# TP=2 also works but uses ~50-60GB per GPU (tighter fit)
TP_SIZE = 4  # Cost-effective: TP=4 works perfectly for 32B model

# Context length: 64K tokens (65536)
# Nomos-1 supports long context (based on Qwen3 architecture)
MAX_MODEL_LEN = 65536  # 64K tokens

# Concurrency / batching: Optimized for 64K context
# Memory calculation: 32B model with TP=4 uses ~16GB/GPU for weights
# With 90% GPU memory utilization (~72GB usable), ~56GB available for KV cache
# With PagedAttention: ~2-4GB per sequence (64K context, typical usage)
# Optimal: 24-28 sequences for good throughput without OOM risk
# Conservative: 16-20 sequences (safer for first deployment)
MAX_NUM_SEQS = 24              # Optimal for 64K context (can increase to 28 if stable, reduce to 20 if OOM)
MAX_NUM_BATCHED_TOKENS = 0      # 0 = auto (let vLLM decide)

# GPU memory utilization: A100-80GB allows good utilization
# For 32B model with 64K context: Can use 0.85-0.90 safely
GPU_MEMORY_UTILIZATION = 0.90   # 90% for A100-80GB (good balance, can reduce to 0.85 if OOM)

# Quantization: FP8 not required for 32B model, but can help with memory
# Nomos-1 may support FP8, but it's not necessary with H200 GPUs
USE_FP8 = False  # Disabled by default (not needed for 32B on H200)

VLLM_PORT = 8000
MINUTES = 60  # seconds
FAST_BOOT = False  # False: Better throughput (models need warmup)

# -----------------------------
# Volumes (for caching)
# -----------------------------

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# -----------------------------
# Secrets (HF token + vLLM API key)
# -----------------------------
# In Modal, create secrets like:
#   modal secret create huggingface-secret HF_TOKEN=hf_...
#   modal secret create vllm-api-secret   VLLM_API_KEY=sk_vllm_...
#
# Inside the container these become environment variables:
#   HF_TOKEN, VLLM_API_KEY

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret",
    required_keys=["HF_TOKEN"],
)

vllm_api_secret = modal.Secret.from_name(
    "vllm-api-secret",
    required_keys=["VLLM_API_KEY"],
)

# -----------------------------
# Base image with vLLM
# -----------------------------

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm>=0.11.2",  # Ensure latest vLLM for Qwen3 MoE support
        "huggingface-hub>=0.36.0",
        "flashinfer-python>=0.5.2",
        # Additional dependencies for Qwen3 models
        "transformers>=4.40.0",
    )
    .env(
        {
            # Optional: speeds up HF downloads on some models
            "HF_XET_HIGH_PERFORMANCE": "1",
            # Enable FP8 support in vLLM (if needed later)
            "VLLM_USE_FP8": "1" if USE_FP8 else "0",
            # PyTorch memory management: reduces fragmentation
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

# -----------------------------
# Modal app & vLLM server
# -----------------------------

app = modal.App("30b-vllm-server")


@app.function(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",  # A100-80GB: Cost-effective, works great with TP=4
    # Alternative: Use H100-80GB for similar performance, or H200-141GB for extra headroom
    scaledown_window=15 * MINUTES,  # Shorter scaledown for smaller model
    timeout=45 * MINUTES,  # Longer timeout for MoE compilation (can take 30+ min on first run)
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[huggingface_secret, vllm_api_secret],
)
@modal.concurrent(max_inputs=32)  # Higher concurrency for smaller model
@modal.web_server(port=VLLM_PORT, startup_timeout=40 * MINUTES)  # Allow more time for MoE compilation (first run can take 30+ min)
def serve():
    """
    Start a vLLM OpenAI-compatible server for Qwen3-30B-A3B-Thinking-2507 (32B MoE).
    
    Optimized for:
      - 64K context length
      - 32B parameter MoE model
      - 4× A100-80GB GPUs with tensor parallelism (TP=4)
      - Cost-effective deployment for experimentation
      - Mathematical reasoning workloads
      - Based on Qwen3-30B-A3B-Thinking-2507 architecture
    """
    import os

    api_key = os.environ["VLLM_API_KEY"]

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]

    cmd += [
        "--served-model-name",
        MODEL_NAME,
        "llm",  # exposed model name for OpenAI-compatible API
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),

        # 🔐 Require API key on all endpoints
        "--api-key",
        api_key,

        # (a) Max context length: 128K tokens
        "--max-model-len",
        str(MAX_MODEL_LEN),

        # (b) GPU memory utilization
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),

        # (c) Concurrency / batching
        "--max-num-seqs",
        str(MAX_NUM_SEQS),

        # (d) Optimization flags
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        
        # (e) Tensor parallelism: TP=8 as recommended by model card
        "--tensor-parallel-size",
        str(TP_SIZE),
    ]
    
    # FP8 quantization: Optional for 32B model (not required with H200)
    if USE_FP8:
        cmd += [
            "--quantization",
            "fp8",  # Use FP8 (F8_E4M3) quantization
        ]
    
    if MAX_NUM_BATCHED_TOKENS > 0:
        cmd += ["--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS)]

    # Performance: Use optimized kernels (not eager mode)
    # Note: For first-time compilation, you might want to use --enforce-eager to avoid hanging
    # But for production, --no-enforce-eager gives better performance
    cmd += ["--no-enforce-eager"]  # Better performance (but slower first compilation)

    print("Starting vLLM with command:")
    print(" ".join(cmd))

    process = subprocess.Popen(" ".join(cmd), shell=True)
    
    # Wait for vLLM server to be ready (but don't block forever)
    # Modal's @web_server needs this function to return so it can start routing traffic
    print("Waiting for vLLM server to start...")
    print(f"Loading {MODEL_NAME} ({N_GPU}x A100-80GB, TP={TP_SIZE}, FP8={'enabled' if USE_FP8 else 'disabled'}, GPU_MEM={GPU_MEMORY_UTILIZATION}, CTX={MAX_MODEL_LEN//1024}K)")
    print("Note: First-time setup includes MoE layer compilation and can take 30+ minutes...")
    max_wait_time = 38 * MINUTES  # Allow up to 38 minutes (within 40 minute startup_timeout)
    check_interval = 5  # Check every 5 seconds
    elapsed = 0
    
    while elapsed < max_wait_time:
        # Check if process is still alive
        if process.poll() is not None:
            raise RuntimeError(f"vLLM process exited with code {process.returncode}")
        
        try:
            # Try to connect to ping endpoint (vLLM's health check)
            req = urllib.request.Request(f"http://localhost:{VLLM_PORT}/ping")
            req.add_header("Authorization", f"Bearer {api_key}")
            
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    print("vLLM server is ready!")
                    # Return immediately - process will keep running in background
                    # Modal will keep the container alive and route traffic to it
                    return
        except (urllib.error.URLError, OSError, Exception):
            # Server not ready yet, continue waiting
            pass
        
        time.sleep(check_interval)
        elapsed += check_interval
        if elapsed % 30 == 0:  # Print status every 30 seconds
            print(f"Still waiting for server... ({elapsed}s elapsed)")
    
    # If we get here, server didn't start in time
    process.terminate()
    raise RuntimeError(f"vLLM server failed to start within {max_wait_time} seconds")


# -----------------------------
# Local test helper
# -----------------------------

@app.local_entrypoint()
async def test(
    api_key: str,
    content: str | None = None,
):
    """
    Simple smoke test.

    Run with:
      modal run modal/30b_server.py --api-key sk_vllm_...

    IMPORTANT: `api_key` must match VLLM_API_KEY in your vllm-api-secret.
    """
    url = serve.get_web_url()

    if content is None:
        content = (
            "Solve the following problem and show your reasoning:\n\n"
            "Let a, b, c be positive real numbers such that abc = 1. "
            "Prove that\n"
            "\\[\n"
            "  \\frac{1}{1+a} + \\frac{1}{1+b} + \\frac{1}{1+c} \\ge 1.\n"
            "\\]"
        )

    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant.",  # Note: Nomos-1 works best without system prompt
    }

    messages: list[dict[str, Any]] = [
        system_prompt,
        {"role": "user", "content": content},
    ]

    headers_auth_only = {
        "Authorization": f"Bearer {api_key}",
    }

    async with aiohttp.ClientSession(base_url=url) as session:
        # Health check with retry logic using /ping endpoint (vLLM's health check)
        print(f"Running health check for server at {url}")
        max_retries = 12
        retry_delay = 10  # seconds
        
        for attempt in range(max_retries):
            try:
                async with session.get("/ping", headers=headers_auth_only, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        print("Health check OK")
                        break
                    else:
                        raise Exception(f"Server returned status {resp.status}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    print(f"Health check failed (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise Exception(f"Server health check failed after {max_retries} attempts: {e}")

        await _send_request(session, "llm", messages, api_key=api_key)


async def _send_request(
    session: aiohttp.ClientSession,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str,
) -> None:
    payload: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {api_key}",
    }

    async with session.post("/v1/chat/completions", json=payload, headers=headers) as resp:
        async for raw in resp.content:
            resp.raise_for_status()
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[len("data: ") :]
            chunk = json.loads(line)
            print(chunk["choices"][0]["delta"].get("content", ""), end="")
    print()


