# Ko-AgentBench
<img width="1200" height="800" alt="banner" src="https://github.com/user-attachments/assets/b65de588-e5ff-4d4b-a386-0e1d0c684cf4" />

English | [한국어](README.md)

<div align="center">
<img src="https://github.com/user-attachments/assets/9cde519b-7935-4e0f-bd34-4d8a81e14103" width="200">

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Ko--AgentBench-blue)](https://huggingface.co/datasets/huggingface-KREW/Ko-AgentBench)
[![Leaderboard](https://img.shields.io/badge/🏆%20Leaderboard-Ko--AgentBench-green)](https://huggingface.co/spaces/huggingface-KREW/Ko-AgentBench)

</div>

---

## Ko-AgentBench ✨

A comprehensive evaluation benchmark for Korean tool-calling agents

Ko-AgentBench evaluates how effectively AI agents leverage tools that Korean users actually use, such as Naver Search, Kakao Map, crypto exchanges, and stock information services.

### 🎯 Key Features

- 🇰🇷 Korea-focused: Naver, Kakao, TMAP, Upbit/Bithumb, LS Securities, Aladin, and more
- 🔑 No API keys required: Run immediately via a cache-based pseudo API
- 🎯 7 independent dimensions: Tool selection, sequential/parallel reasoning, error handling, efficiency, long context, etc.
- 🔄 Reproducible: Repeatable runs under the same conditions for research reproducibility

> [!TIP]
> Why Ko-AgentBench?
>
> Limitations of existing benchmarks
> - Most tool-calling benchmarks are English-centric and don’t reflect the Korean environment or real Korean user scenarios.
> - They often only check “was the correct API called?” and miss real workflows (error handling, efficiency, context maintenance).
>
> What sets Ko-AgentBench apart
> - Built on real Korean tools: Naver Search/Blog, Kakao Map/Place Search, TMAP routing, Upbit/Bithumb crypto, LS Securities stock info, KTO festivals, Aladin book search, etc.
> - Realistic multi-step scenarios: Not just a single API call; reflects real workflows chaining multiple tools and passing data.
> - Comprehensive evaluation: Based on seven principles—realism, clarity, discriminability, robustness, efficiency, reproducibility, and extensibility.

### 💡 Cache System: Start instantly without API keys

Ko-AgentBench ships with pre-collected API response caches so you can run the benchmark without real API calls.

- Read mode (default): Uses cache only, no API keys → anyone can evaluate right away
- Write mode: Makes real API calls and saves to cache → use for extending datasets

```bash
# Run without API keys (cache mode)
uv run run_benchmark_with_logging.py --levels L1 --model openai/gpt-4

# Make real API calls (API keys required)
uv run run_benchmark_with_logging.py --cache-mode write
```

---

## 🛠️ Available API Tools

Ko-AgentBench provides various Korean service APIs that users actually use in daily life.

| Service | Tools | Description |
| :--- | :--- | :--- |
| **Naver Search** | `Search_naver_web`<br>`Search_naver_blog`<br>`Search_naver_news` | Naver integrated search, blog, and news search APIs |
| **Kakao Local** | `AddressToCoord_kakao`<br>`CoordToAddress_kakao`<br>`PlaceSearch_kakao`<br>`CategorySearch_kakao` | Address-coordinate conversion, place search, category search |
| **Upbit** | `CryptoPrice_upbit`<br>`MarketList_upbit`<br>`CryptoCandle_upbit` | Crypto current price, market list, candle chart data |
| **Bithumb** | `CryptoPrice_bithumb`<br>`OrderBook_bithumb`<br>`MarketList_bithumb`<br>`CryptoCandle_bithumb` | Crypto current price, order book, market list, candle chart |
| **LS Securities** | `StockPrice_ls`<br>`MarketIndex_ls`<br>`OrderBook_ls`<br>`SectorStock_ls`<br>`StockTrades_ls` | Domestic/international stock prices, market index, order book, sector stocks, trade history |
| **Korea Investment** | `StockPrice_kis`<br>`USStockPrice_kis`<br>`StockChart_kis` | Domestic stock price, US stock price, chart data |
| **Aladin** | `ItemSearch_aladin`<br>`ItemList_aladin`<br>`ItemLookup_aladin` | Book search, bestseller/new release lists, book details |
| **TMAP** | `POISearch_tmap`<br>`Geocoding_tmap`<br>`ReverseGeocoding_tmap`<br>`CarRoute_tmap`<br>`CategorySearch_tmap` | POI search, address-coordinate conversion, car routing, category search |
| **Naver Maps** | `Directions_naver` | Public transit/car/walking directions |
| **Korea Tourism** | `FestivalSearch_kto` | National festival information search |

> **Note**: All APIs operate in cache mode; in Read mode (default), no actual API keys are required.

---

## 📊 The 7 Independent Evaluation Dimensions

Tool-calling ability isn’t one-dimensional. “Using tools well” involves choosing the right tool, planning to chain tools, handling errors, and operating efficiently. Ko-AgentBench separates these abilities into seven independently measured dimensions—they are not difficulty tiers but orthogonal capabilities.

| Level | Task | Description |
| :--- | :--- | :--- |
| **L1** | Single tool call | Verify executing a given single tool with correct parameters |
| **L2** | Tool selection | Choose the most suitable tool among multiple candidates |
| **L3** | Sequential reasoning | Plan and execute sequences where one tool’s output feeds into the next |
| **L4** | Parallel reasoning | Execute multiple tools concurrently and synthesize results |
| **L5** | Error handling & robustness | Handle API failures, missing info, and other exceptional situations |
| **L6** | Efficient tool use | Reuse past tool results to avoid redundant calls |
| **L7** | Long-context memory | Maintain and utilize long multi-turn context to call tools appropriately |

Each dimension is measured independently; the composite score is a weighted average, allowing fine-grained diagnosis of strengths and weaknesses.

---

## 🚀 Quickstart

### 1) Install
```bash
# Clone repo
git clone https://github.com/Hugging-Face-KREW/Ko-AgentBench
cd Ko-AgentBench

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2) Set LLM API keys

Only the LLM’s API key is required (tool API keys are unnecessary in cache mode).

```bash
# LLM Model API key (required)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
export OPENROUTER_API_KEY="your-openrouter-key"
# Optional headers recommended by OpenRouter
export OPENROUTER_APP_URL="https://your-service.tld"
export OPENROUTER_APP_TITLE="Ko-AgentBench Runner"
```

### 3) Run and Evaluate
```bash
# Run benchmark (L1, cache read mode)
uv run run_benchmark_with_logging.py --levels L1 --model openai/gpt-4

# OpenRouter example
uv run run_benchmark_with_logging.py --levels L1 --model openrouter/anthropic/claude-3.5-sonnet

# Evaluate (enter execution date as YYYYMMDD)
uv run evaluate_model_run.py --date 20251022 --model openai/gpt-4 --format all

# (Optional) Validate v2 task schemas
uv run python scripts/validate_task_schemas.py

# Build and validate v2 A+B sample tasks (20)
uv run python scripts/build_v2_sample_tasks.py
uv run python scripts/validate_task_schemas.py --tasks-dir bench/tasks_v2 --strict
```

---

## 📁 Project Structure

### Folder layout
```
Ko-AgentBench/
├─ bench/
│  ├─ tasks/       # YAML task definitions
│  ├─ tools/       # Tool specs and adapters
│  ├─ runner/      # Engine and metrics
│  └─ cache/       # API response cache
├─ logs/           # Execution logs
├─ reports/        # Evaluation reports
├─ configs/        # Configuration files
├─ run_benchmark_with_logging.py
├─ evaluate_model_run.py
└─ README.md
```

### Pipeline

```
[1] Run: run_benchmark_with_logging.py
    └─ Execution logs → logs/
         (conversation, tool calls, parameters, cache hit rate, errors)

[2] Evaluate: evaluate_model_run.py
    └─ Reports → reports/{model}_{date}/
         (metrics, JSON/CSV/Markdown formats)
```

Run and evaluation are decoupled for reproducibility and automation.

---

## ⚡ Running the Benchmark

### Basic usage
Use `run_benchmark_with_logging.py` to evaluate a model.

```bash
# Run all levels (cache read mode)
uv run run_benchmark_with_logging.py

# Run specific levels
uv run run_benchmark_with_logging.py --levels L1,L2,L3

# Specify a model
uv run run_benchmark_with_logging.py --model openai/gpt-4

# Use a local model
uv run run_benchmark_with_logging.py --use-local --model Qwen/Qwen2.5-7B-Instruct

# Make real API calls and write cache
uv run run_benchmark_with_logging.py --cache-mode write

# Repeat each task 3 times
uv run run_benchmark_with_logging.py --repetitions 3

# (L6/L7) Run in seed replay mode (default is full_rollout)
uv run run_benchmark_with_logging.py --seed-replay
```

### Key options
**Dataset selection**
- `--levels`: Levels to run (e.g., `L1,L2,L6,L7`) — default: all
- `--release-scope`: `AB` (default, L1-L7) or `ALL` (all detected levels)
- `--data-dir`: Task dataset directory (default: `bench/tasks`, v2 sample: `bench/tasks_v2`)

**Model settings**
- `--model`: Model ID (e.g., `openai/gpt-4`, `anthropic/claude-3-5-sonnet-20241022`)
- `--use-local`: Use local Transformers
- `--quantization`: `4bit`/`8bit`
- `--device`: `cuda`/`cpu`/`auto`
- `--dtype`: `auto`/`float16`/`bfloat16`/`float32`

**Execution control**
- `--max-steps`: Max steps per task (default: 10)
- `--timeout`: Timeout per task in seconds (default: 60)
- `--max-retries`: Retry attempts for failed LLM calls (default: 3, increase for 429 mitigation)
- `--repetitions`: Repeated runs (default: 3, for Pass@k/stability@k)
- `--seed-replay`: Use seed replay for L6/L7 (default is `full_rollout`)
- `--skip-task-schema-validation`: Disable v2 task schema validation
- `--strict-task-schema`: Fail fast when v2 schema validation fails
- `--schema-mode`: Task schema selection (`auto`/`v2`/`v1`, default: `auto`)
- `--no-save-logs`: Disable log saving

**Cache mode**
- `--cache-mode`:
    - `read` (default): Use saved cache only
    - `write`: Make real API calls and save cache (requires API keys in `configs/secrets`)

**Result location**
- Default: `logs/benchmark_results/by_model/{model}/{timestamp}/`

---

## 📏 Evaluation

### Usage
Use `evaluate_model_run.py` to analyze logs and generate reports.

```bash
# Basic evaluation
uv run evaluate_model_run.py --date 20251022 --model azure/gpt-4o

# Quick test (1 per level)
uv run evaluate_model_run.py --date 20251022 --model azure/gpt-4o --quick
```

### Key options
- `--date`: Benchmark execution date (YYYYMMDD)
- `--model`: Model ID to evaluate
- `--release-scope`: `AB` (default, L1-L7) or `ALL`
- `--judge-models`: Judge model(s) (Default: 3-model ensemble)
- `--sample N`: Evaluate N samples per level
- `--quick`: 1 sample per level (sampling)
- `--bootstrap-samples`: Number of bootstrap samples for CI (default: 1000)
- `--no-bootstrap-ci`: Disable bootstrap CI calculation
- `--judge-max-retries`: Judge API retry attempts (default: 5, for 429 mitigation)
- `--judge-backoff-base`: Initial backoff seconds for Judge retries (default: 2.0)
- `--judge-backoff-cap`: Maximum backoff seconds for Judge retries (default: 120.0)
- `--strict-v2-eval-schema`: Treat v2 schema validation failures as errors when exporting `v2-jsonl`
- `--format`: Output format (`json`/`csv`/`markdown`/`v2-jsonl`/`all`)

### Output location
- Output: `reports/{model}_{date}/`
    - `evaluation_report.json`
    - `evaluation_summary.csv`
    - `evaluation_report.md`
    - `evaluation_v2.jsonl` (when `--format v2-jsonl` or `--format all`)

---

## 🧩 Evaluation Levels and Tasks

Seven levels assess the agent’s tool-calling capabilities.

| Level | Area | Example | Key metrics |
|-------|------|---------|-------------|
| **L1** | Single tool call | “How many minutes by car from Pangyo Station to Jamsil Baseball Stadium?” | ToolAcc, ArgAcc, CallEM, RespOK |
| **L2** | Tool selection | “I want to check the current order book for POSCO Holdings” | SelectAcc |
| **L3** | Sequential reasoning | “Find universities near Cheongnyangni Station, then count nearby hospitals” | FSM, PSM, ΔSteps_norm |
| **L4** | Parallel reasoning | “Fetch BTC prices from multiple exchanges and compare” | Coverage, SourceEPR |
| **L5** | Error handling & robustness | “Search the iPhone 17 release date” (fallback when API fails) | AdaptiveRoutingScore, FallbackSR |
| **L6** | Efficient tool use | “Find books on Python algorithmic trading” (avoid duplicate calls) | RedundantCallRate, EffScore |
| **L7** | Long-context memory | “I’ve been into Bitcoin lately…” (multi-turn dialogue) | ContextRetention, RefRecall |

---

## 🧩 Metrics

### Common metrics (all levels)
| Metric | Description | Calculation |
|---|---|---|
| **SR** (Success Rate) | Task completion score | LLM Judge 1–5 → (score-1)/4 |
| **EPR/CVR** | Effective call ratio | Valid calls / total calls |
| **Pass@k** | At least one success in k tries | Mean of task-level any-success (0/1) |
| **Stability@k** | Repetition stability | Successful attempts / k |

### Level-specific metrics

**L1: Single tool call**
| Metric | Description | Calculation |
|---|---|---|
| **ToolAcc** | Correct tool selection | match=1, else 0 |
| **ArgAcc** | Argument accuracy | LLM Judge 1–5 → 0–1 |
| **CallEM** (Call Exact Match) | Tool+args exact match | 0 or 1 |
| **RespOK** | Response format compliance | 0 or 1 |

**L2: Tool selection**
| Metric | Description | Calculation |
|---|---|---|
| **SelectAcc** | Correct selection rate | 0 or 1 |

**L3: Sequential reasoning**
| Metric | Description | Calculation |
|---|---|---|
| **FSM** (Full Sequence Match) | Exact call order match | 0 or 1 |
| **PSM** (Partial Sequence Match) | Required tools coverage | required included / total required |
| **ΔSteps_norm** | Efficiency vs. shortest path | min(1, min steps / actual steps) |

**L4: Parallel reasoning**
| Metric | Description | Calculation |
|---|---|---|
| **Coverage** | Required tool execution rate | successful required / total required |
| **SourceEPR** | Avg. valid-call rate per tool | mean(valid / total) |

**L5: Error handling & robustness**
| Metric | Description | Calculation |
|---|---|---|
| **AdaptiveRoutingScore** | Agility of switching to fallback | 1 / (1 + switch delay steps) |
| **FallbackSR** | Fallback success rate | fallback success / attempts |

**L6: Efficient tool use**
| Metric | Description | Calculation |
|---|---|---|
| **RedundantCallRate** | Redundant call avoidance | 1 - (redundant / reuse opportunities) |
| **EffScore** | Efficiency when successful | min(1, min steps / actual steps) |

**L7: Long-context memory**
| Metric | Description | Calculation |
|---|---|---|
| **ContextRetention** | Context maintenance ability | LLM Judge 1–5 → 0–1 |
| **RefRecall** | Recall of prior info | LLM Judge 1–5 → 0–1 |

### Judge evaluation
- Judge ensemble (default): GPT-5.1, Claude Sonnet 4.5, Gemini 2.5 Pro
- Aggregation: mean or median
- Blind evaluation without model names for fairness

---

## 📊 Leaderboard

Results are saved to `reports/{model}_{date}/`:
- JSON (`evaluation_report.json`)
- CSV (`evaluation_summary.csv`)
- Markdown (`evaluation_report.md`)

---

## Usage examples

```bash
# 1) Evaluate L1–L3 with GPT-4
uv run run_benchmark_with_logging.py --levels L1,L2,L3 --model openai/gpt-4

# 2) Full levels with Claude + write cache
uv run run_benchmark_with_logging.py --model anthropic/claude-3-5-sonnet-20241022 --cache-mode write

# 3) Local model with 4-bit quantization
uv run run_benchmark_with_logging.py --use-local --model Qwen/Qwen2.5-7B-Instruct --quantization 4bit --device cuda

# 4) Multi-turn dialogue levels
uv run run_benchmark_with_logging.py --levels L6,L7 --max-steps 20

# 5) Generate evaluation reports (default: single Judge)
uv run evaluate_model_run.py --date 20251022 --model azure/gpt-4o --format all

# 6) Quick sampled evaluation
uv run evaluate_model_run.py --date 20251022 --model azure/gpt-4o --quick
```

---

## ⚖️ License

Licensed under [Apache-2.0](LICENSE).
