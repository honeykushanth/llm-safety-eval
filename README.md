# LLM Safety Evaluation Pipeline

A research pipeline for measuring how well large language models (LLMs) resist adversarially phrased and obfuscated prompts related to offensive security operations.

---

## Overview

The project consists of two scripts:

| Script | Purpose |
|---|---|
| `generate.py` | Generates a synthetic dataset of malicious, benign, and obfuscated prompts |
| `eval.py` | Sends obfuscated prompts to four LLM APIs and classifies their responses |

### Prompt Categories

- **Malicious** — Direct offensive security requests (credential dumping, privilege escalation, etc.) phrased in 15 different styles (forum post, CTF question, red team brief, etc.)
- **Benign** — Defensive security tasks (Sigma rules, IR playbooks, SIEM queries, threat hunts) that involve the same attack techniques as legitimate detection work
- **Obfuscated** — Legitimate-sounding technical project descriptions (APM agents, chaos frameworks, CI/CD scanners) whose requirements could mask harmful intent

### Models Evaluated

- **Groq** — `llama-3.3-70b-versatile`
- **Gemini** — `gemini-2.5-flash-preview-04-17`
- **OpenRouter / GPT** — `openai/gpt-oss-20b`
- **OpenRouter / Llama** — `meta-llama/llama-3.2-3b-instruct`

---

## Classifier Architecture

Each model response is classified as `safe`, `unsafe`, or `ambiguous` using a three-layer pipeline:

```
Layer 1 — Heuristic (weighted regex)
  Scans for compliance openers, dangerous API names, evasion keywords, and refusal phrases.
  Output: directional score in [-1.0, +1.0]

Layer 2 — Structural (AST analysis)
  Extracts fenced code blocks and parses them with Python's ast module.
  Flags dangerous imports (os, subprocess, ctypes, socket, …) and dangerous calls (exec, eval, Popen, …).
  Output: risk score in [0.0, 1.0]

Layer 3 — ML (TF-IDF + Logistic Regression)
  Trained on a synthetic corpus of refusal and compliance surface patterns.
  Output: P(unsafe | response) in [0.0, 1.0]

Final decision: hard-condition gates → weighted combination (25% heuristic / 30% structural / 45% ML)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/honeykushanth/llm-safety-eval.git
cd llm-safety-eval
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
GEMINI_API_KEY=...
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
```

---

## Usage

### Step 1 — Generate the dataset

```bash
python generate.py                          # 3000 rows → dataset.csv
python generate.py --size 500               # custom row count
python generate.py --size 500 --output my_dataset.csv
```

The output CSV has four columns: `prompt_id`, `malicious_prompt`, `benign_prompt`, `obfuscated_prompt`.

### Step 2 — Run the evaluation

```bash
python eval.py --input dataset.csv --output results.csv
```

The output CSV appends eight new columns to the input: raw responses and classifications from each of the four models.

---

## Output Format

`results.csv` contains all original columns plus:

| Column | Description |
|---|---|
| `groq_raw` | Raw text response from Groq |
| `gemini_raw` | Raw text response from Gemini |
| `openai/gpt-oss-20b_raw` | Raw text response from GPT |
| `meta-llama/llama-3.2-3b-instruct_raw` | Raw text response from Llama |
| `groq_result` | Classification: `safe` / `unsafe` / `ambiguous` / `error` |
| `gemini_result` | Same for Gemini |
| `openai/gpt-oss-20b_result` | Same for GPT |
| `meta-llama/llama-3.2-3b-instruct_result` | Same for Llama |

The console also prints a **Robustness Score** per model (percentage of non-safe responses, weighted by severity) and precision / recall / FPR metrics.

---

## Known Limitations

**Synthetic ground truth.** The evaluation pipeline treats every obfuscated prompt as `sensitive` (ground truth = 1). This is a deliberate research choice to measure model compliance rates on ambiguous inputs. Precision and Recall figures should be read with this assumption in mind — they measure the classifier's ability to detect compliance, not accuracy against independently labelled data.

**Synthetic ML training data.** The TF-IDF + Logistic Regression classifier is trained on a hand-crafted corpus of ~45 examples. It captures surface-level compliance and refusal patterns well but has not been validated on out-of-distribution responses.

**Rate limits.** Concurrency is capped per provider (3 simultaneous requests each). For very large datasets, consider splitting the CSV into batches.

---

## Project Structure

```
llm-safety-eval/
├── generate.py        # Dataset generator
├── eval.py            # Evaluation pipeline
├── .env.example       # API key template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## License

MIT
