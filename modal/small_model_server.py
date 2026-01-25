# Qwen3-4B vLLM Server Configuration
# Optimized for 4B parameter model finetuned for generator-critique workflow
# Model: hmdmahdavi/olympiad-curated-qwen3-4b-thinking-generator-critique

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

# Qwen3-4B: 4B parameter model finetuned for olympiad problem solving
# Specialized for generator-critique workflow
MODEL_NAME = "hmdmahdavi/olympiad-curated-qwen3-4b-thinking-distill-30b"
MODEL_REVISION = None  # Use latest revision

# Hardware: 4B model fits easily on a single A100-80GB
# Memory calculation: 4B params = ~8GB total + KV cache (~10-15GB for 128K context) = ~18-23GB
# Single A100-80GB has plenty of headroom
N_GPU = 1  # Single A100-80GB GPU

# Tensor parallelism: Not needed for 4B model (fits on single GPU)
TP_SIZE = 1  # No tensor parallelism needed

# Long context (≈65K) config
MAX_MODEL_LEN = 65536  # close to 65K target, and power-of-two friendly

# Concurrency / batching: Optimized for 65K context on single GPU
# Memory calculation: 4B model uses ~8GB for weights
# With 88% GPU memory utilization (~70GB usable), ~62GB available for KV cache
# Reduced to 6 to prevent OOM: fewer concurrent sequences = less KV cache pressure
# With 65K tokens per sequence, KV cache can be 10-20GB per sequence
# 6 sequences max = 60-120GB KV cache worst case, but typically much less
MAX_NUM_SEQS = 8             # Reduced to prevent OOM from concurrent long sequences
MAX_NUM_BATCHED_TOKENS = 0 # Cap total tokens in flight (~33K per sequence avg, allows 2-3 long sequences)

# GPU memory utilization: Reduced to prevent OOM with long sequences
GPU_MEMORY_UTILIZATION = 0.88   # use 88% of VRAM (more headroom for KV cache growth)

# Quantization: Not needed for 4B model (fits easily on A100-80GB)
USE_FP8 = False  # Disabled by default (not needed for 4B model)

VLLM_PORT = 8000
MINUTES = 60  # seconds
FAST_BOOT = False  # True: faster cold start, False: better throughput after warmup

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
        "vllm==0.11.2",  # Pin version to match old working config
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
    )
    .env(
        {
            # Optional: speeds up HF downloads on some models
            "HF_XET_HIGH_PERFORMANCE": "1",
            # PyTorch memory management: reduces fragmentation (critical for preventing OOM)
            # Note: Both env vars needed for different PyTorch versions
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

# -----------------------------
# Modal app & vLLM server
# -----------------------------

app = modal.App("4b-vllm-server")


@app.function(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",  # Single A100-80GB: Perfect for 4B model
    scaledown_window=5 * MINUTES,  # Shorter scaledown for smaller model
    timeout=30 * MINUTES,  # Timeout for long-running inference
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[huggingface_secret, vllm_api_secret],
)
@modal.concurrent(max_inputs=6)  # Must match MAX_NUM_SEQS
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)  # Startup timeout
def serve():
    """
    Start a vLLM OpenAI-compatible server for Qwen3-4B finetuned model.
    
    Optimized for:
      - 128K context length
      - 4B parameter model
      - Single A100-80GB GPU (no tensor parallelism needed)
      - Cost-effective deployment for generator-critique workflow
      - Mathematical reasoning workloads
      - Based on Qwen3-4B architecture
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
        
        # (e) Tensor parallelism: TP=1 (no parallelism needed for 4B model)
        "--tensor-parallel-size",
        str(TP_SIZE),
    ]
    
    # FP8 quantization: Optional for 4B model (not needed with A100-80GB)
    if USE_FP8:
        cmd += [
            "--quantization",
            "fp8",  # Use FP8 (F8_E4M3) quantization
        ]
    
    if MAX_NUM_BATCHED_TOKENS > 0:
        cmd += ["--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS)]

    # Fast boots vs max performance
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    print("Starting vLLM with command:")
    print(" ".join(cmd))

    process = subprocess.Popen(" ".join(cmd), shell=True)
    
    # Wait for vLLM server to be ready (but don't block forever)
    # Modal's @web_server needs this function to return so it can start routing traffic
    print("Waiting for vLLM server to start...")
    max_wait_time = 9 * MINUTES  # Allow up to 9 minutes (within 10 minute startup_timeout)
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
      modal run modal/small_model_server.py --api-key sk_vllm_...

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
        "content": "You are a helpful assistant specialized in solving mathematical olympiad problems.",
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

