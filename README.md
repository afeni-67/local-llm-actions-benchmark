# Local LLM Benchmark on GitHub Actions

Benchmark a **small open-weight LLM running entirely on a GitHub-hosted Ubuntu runner** (CPU only).  
No paid LLM APIs. No remote inference endpoints.

## Model

| Item | Value |
|------|--------|
| **Model** | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) (4B parameters) |
| **Weights used** | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) → `Qwen3-4B-Q4_K_M.gguf` |
| **Quantization** | **Q4_K_M** (GGUF, ~2.5 GB on disk) |
| **License** | **Apache License 2.0** (see model card on Hugging Face) |
| **Runtime** | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) wrapping **llama.cpp**, **`n_gpu_layers=0`** (CPU only) |

**Why 4B not 8B?** Standard public `ubuntu-latest` runners have **4 vCPU / 16 GB RAM / 14 GB SSD**. Q4_K_M 4B fits comfortably; 8B uses more RAM and is slower on CPU, risking timeouts.

### Expected resources

- **Disk:** ~3 GB for the GGUF + Python deps  
- **RAM:** plan for **~4–8 GB** RSS while loaded (context small: 1024)  
- **CPU:** multi-thread via `LLM_THREADS` (default 3 on Actions)

### How the model is downloaded

`huggingface_hub.hf_hub_download` pulls  
`Qwen/Qwen3-4B-GGUF` / `Qwen3-4B-Q4_K_M.gguf` into `models/` at job time.  
No API key is required for this public file.

## Task

Synthetic **support-ticket classification**. For each record the model must return **strict JSON**:

```json
{"category":"billing|technical|shipping|account|product|other","sentiment":"negative|neutral|positive","confidence":0.0}
```

Python validates every response (`validate_output.py`).

## Dataset

- `generate_test_data.py` builds **≥ 1000** synthetic tickets (`data/synthetic_tickets.jsonl`).
- Fully synthetic — no real customer data.
- To use a real dataset later: replace the JSONL with objects that have at least `id` and `text` fields; keep the same prompt/schema or adjust `model_runner.py`.

## GitHub Actions limits (relevant)

Sources: [GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners), usage limits.

| Limit | Value |
|--------|--------|
| Public repo standard Linux | **4 vCPU, 16 GB RAM, 14 GB SSD** (`ubuntu-latest`) |
| Max **job** time (hosted) | **6 hours (360 minutes)** hard cap |
| Minutes billing | **Free** for standard runners on **public** repos |
| GPU | **Not used** (and not available on standard runners) |

This workflow sets `timeout-minutes: 360`. A full 1000-record run may take from tens of minutes to a few hours depending on CPU speed; the summary **extrapolates** records/minute to a theoretical 6-hour capacity (not a guarantee).

## Metrics reported

- total / successful / failed records  
- records/minute (wall clock)  
- average and total inference time  
- approximate peak RSS (process)  
- estimated records in a 6-hour job (`rpm × 360`)

Artifacts: `benchmark_summary.json`, `benchmark_results.csv`.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_test_data.py --count 1000
python benchmark.py --limit 20          # smoke test
python benchmark.py                     # full 1000
```

## Run on GitHub Actions

1. Push this repo to GitHub (**public** recommended for free unlimited standard minutes).  
2. **Actions** → **Local LLM Benchmark** → **Run workflow**.  
3. Optional input: `limit` (e.g. `50` for a quick smoke test, `1000` for full).  
4. Download the **benchmark-results** artifact when finished.

No repository secrets are required for the default public GGUF download.

## Interpreting records/minute

`records_per_minute` = completed records / wall-clock minutes including Python overhead.  
Use it to compare machines or quantizations.  
`estimated_records_in_6h_job` = that rate × 360; real long runs may be slower (thermal throttling, log noise, etc.).

## What “practical for large batches” looks like

As a rule of thumb on this runner class:

- **Smoke OK:** model loads, success rate **> 80%** JSON validity on 50+ records  
- **Useful batching:** **≥ ~5–10 records/minute** sustained with valid JSON (order-of-magnitude; measure your own run)  
- **Not practical here:** model OOM, success rate near 0, or &lt; 1 record/minute for short prompts  

Always trust **your** Actions log and artifact — never invented numbers.

## License

Benchmark code: MIT (or as designated by the repo owner).  
Model weights: **Apache-2.0** (Qwen3-4B / GGUF distribution on Hugging Face).
