


# 🚀 TurboQuant vs RotorQuant: Infinite Context on 8GB GPUs

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-Fast%2B-purple)](https://github.com/astral-sh/uv)
[![Llama.cpp](https://img.shields.io/badge/Engine-Llama.cpp-green)](#)

Welcome to the official live benchmark and demo for the **TurboQuant vs RotorQuant** YouTube video. 

This repository contains the setup guide and Python benchmarking script to run a **100,000-token context window** entirely locally on a budget 8GB consumer GPU, by compressing the KV Cache by 10x using 3-bit quantization.

## 🧠 The TL;DR: What are we testing?

Running massive contexts normally causes an `Out of Memory` (OOM) error because the AI's "notebook" (the KV Cache) balloons to over 5GB. 
*   **TurboQuant (`turbo3`)**: Compresses the cache to 3-bits using a dense 128x128 Walsh-Hadamard Transform (WHT). It saves memory but is extremely mathematically heavy (16,384 ops).
*   **RotorQuant (`iso3`)**: Replaces the dense matrix with isolated **4D Quaternion block-diagonals**. It achieves the exact same 10x memory savings but requires **44x less math** (372 ops).

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:
1. `git` and `cmake` (for building the engine)
2. `uv` (The blazing-fast Python package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
3. A C++ compiler (GCC/Clang/MSVC)

---

## ⚙️ Step-by-Step Setup Guide

### 1. Initialize the Python Environment
Clone this repo (or set up a folder) and install the dependencies in milliseconds using `uv`:
```bash
mkdir ai-demo && cd ai-demo
uv init
uv add openai requests rich
```

### 2. Compile the RotorQuant Engine
We need to clone the community-optimized `llama.cpp` fork containing the RotorQuant architecture.
```bash
git clone https://github.com/johndpope/llama-cpp-turboquant.git rotorquant-engine
cd rotorquant-engine
git checkout feature/planarquant-kv-cache
```

Compile it for your specific hardware. Choose **ONE** of the following:

**🟩 NVIDIA GPU (RTX 3060, 4060, etc.) - *Recommended***
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```
**🟦 CPU Only (Servers, standard laptops)**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```
**🍎 Mac Apple Silicon (M1/M2/M3)**
```bash
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.logicalcpu)
```

### 3. Download the Model
We will use **Qwen 3.5 2B**, a highly capable and lightweight model.
```bash
cd ..
mkdir models
wget -P models/ https://huggingface.co/Qwen/Qwen3.5-2B-Instruct-GGUF/resolve/main/qwen3.5-2b-instruct-q4_k_m.gguf
```

---

## 🏁 Running the Benchmark

To run the head-to-head race, we need to spin up two servers simultaneously. Open two separate terminal windows.

**Terminal 1: Start the TurboQuant Server (Port 8082)**
> *Note: If you are strictly on CPU, change `-ngl 99` to `-ngl 0`*
```bash
./rotorquant-engine/build/bin/llama-server \
  -m models/qwen3.5-2b-instruct-q4_k_m.gguf \
  -c 100000 \
  -ngl 99 \
  --cache-type-k turbo3 \
  --cache-type-v turbo3 \
  --port 8082
```

**Terminal 2: Start the RotorQuant Server (Port 8081)**
```bash
./rotorquant-engine/build/bin/llama-server \
  -m models/qwen3.5-2b-instruct-q4_k_m.gguf \
  -c 100000 \
  -ngl 99 \
  --cache-type-k iso3 \
  --cache-type-v iso3 \
  --port 8081
```

**Terminal 3: Run the Live Demo**
Ensure `demo.py` is in your directory (from the video guide), and execute it using `uv`:
```bash
uv run demo.py
```

Watch the terminal as it downloads the massive C++ codebase, streams it to both servers, and benchmarks the speeds side-by-side using a beautiful `rich` UI! Check your Task Manager / `nvidia-smi`—your VRAM will stay safely around `~2.3GB` despite processing 100,000 tokens!

---

## 📊 Understanding the Results (The "Caching" Easter Egg)

If you run `demo.py` **twice in a row**, you will notice something fascinating. 

**Run 1 (The True Math Race):** 
You will see the true processing times. TurboQuant currently benefits from 30 years of dense-matrix hardware optimizations (like NVIDIA `cuBLAS`), allowing it to muscle through the raw math quickly. RotorQuant's 4D Quaternion math is brand new and waiting for custom CUDA kernels.

**Run 2 (The Decompression Race):**
If you run the script a second time, the server uses **Prompt Caching**. It doesn't recalculate the math; it just reads the saved KV Cache from your RAM.
You will see times drop to fractions of a second:
*   🐌 **TurboQuant:** ~0.14s
*   ⚡ **RotorQuant:** ~0.05s

**Why does RotorQuant win?** Even when cheating with a saved cache, the AI must *decompress* the data to read it. Reversing TurboQuant's dense 128x128 global web takes massive compute. Reversing RotorQuant's isolated 4D pods is nearly instant. **RotorQuant is consistently 3x faster at reading cached data!**

---

### 📺 Watch the Full Breakdown
If you found this repository before the video, check out the full visual explanation of the math here: **[Insert YouTube Video Link Here]**
