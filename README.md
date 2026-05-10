# Counterexample-Driven Medical Agent for Medical Multiple Choice QA

This Agent is an evaluation-oriented framework for studying whether adversarial self-auditing can improve the reliability of large language models on medical multiple-choice questions. The system implements a set of baseline and ablation pipelines around a common clinical reasoning task: given a question stem and answer choices, predict the correct answer while optionally stress-testing the initial prediction with counterexamples, rubric-based adjudication, abstention, and repair.

The codebase is a FastAPI service for single-example inference and asynchronous dataset evaluation, records structured traces for every model call, and reports accuracy, selective accuracy, abstention, token usage, latency, and repair/breakage statistics.

This repository is designed for research and benchmarking. It is not a clinical decision-support system and must not be used to provide medical advice.

## Research objective

Medical question answering systems often produce plausible reasoning even when the final answer is wrong. A central question in this project is whether an LLM can be made more robust by separating the reasoning process into specialized roles:

1. a solver proposes an answer and grounds salient findings in the question text;
2. a counterexample generator searches for an objective flaw or stronger alternative;
3. a referee applies a conservative decision rule to keep, revise, or abstain;
4. a repair stage produces the final answer only when revision is justified.

The framework supports direct comparisons against standard prompting baselines and several ablations. This makes it suitable for experiments that ask not only whether accuracy improves, but also where the improvement comes from and what additional cost it incurs.

## Method overview

The full pipeline begins with a structured solver response. The solver selects an answer, extracts key features, quotes supporting evidence from the question, and records top alternatives. A counterexample agent then attempts to challenge the solver using only explicit facts from the question. Counterexamples are typed according to a fixed taxonomy:

- `COMPETING_DDX`: an alternative diagnosis or answer better explains the case.
- `FEATURE_IGNORED`: the solver missed a critical stated finding.
- `CONTRADICTION`: the solver answer conflicts with the stem.
- `EINSTELLUNG_TRAP`: the solver appears anchored on a familiar pattern despite contrary evidence.
- `AMBIGUITY_EXPLOIT`: the question is underdetermined or supports more than one plausible answer.

A referee evaluates the solver and counterexample using a conservative policy. The default behavior is to keep the solver answer unless the counterexample identifies an objective flaw grounded in the question. The rubric scores fact consistency, coverage of findings, clinical plausibility, and parsimony on a 0-3 scale. The referee may return `KEEP`, `REVISE`, or `ABSTAIN`. If revision is warranted, the repair agent writes the final response while addressing the counterexample explicitly.

The implementation includes guardrails for common LLM output failures: strict JSON parsing, a formatter retry pass, bounded integer coercion, choice-index validation, quote sanitization, attack-type normalization, and consistency checks between referee decisions and winners.

## Pipeline variants

The `variant` field controls the experimental condition.

| Variant | Description |
| --- | --- |
| `B0_DIRECT` | Single direct answer call. No chain-of-thought-style structured solver, critique, or repair. |
| `B1_COT` | Structured solver with key features and short rationale. No counterexample or referee. |
| `B2_SELF_CONSISTENCY` | Repeated direct-answer sampling with majority vote. Controlled by `self_consistency_k`. |
| `B3_SELF_REFINE` | Solver followed by self-critique and repair when a critique suggests an alternative. |
| `A0_FULL` | Main pipeline: solver, counterexample, rubric referee, abstention support, and repair. |
| `A1_SELF_REFINE_ONLY` | Alias of the self-refine pipeline used as an ablation against the counterexample architecture. |
| `A2_REFEREE_NO_RUBRIC` | Full pipeline with a free-form conservative referee instead of the explicit scoring rubric. |
| `A3_NO_REPAIR` | Full pipeline without the repair stage; revision directly selects the counterexample alternative. |
| `A4_RESTRICTED_ATTACKS` | Full pipeline with counterexample generation restricted to selected attack types. Defaults to `CONTRADICTION` and `FEATURE_IGNORED`. |
| `A5_MULTI_COUNTEREXAMPLES` | Generates multiple counterexamples, adjudicates the set, selects the strongest counterexample, and repairs if needed. Controlled by `multi_counter_k`. |

## Repository layout

```text
.
├── app
│   ├── api
│   │   └── routes.py              # FastAPI endpoints for health, pipeline inference, and evaluation jobs
│   ├── core
│   │   ├── config.py              # Environment-backed runtime settings
│   │   ├── run_registry.py        # In-memory evaluation job status registry
│   │   └── util.py                # JSON, hashing, timestamps, and filesystem helpers
│   ├── llm
│   │   ├── client.py              # OpenAI-compatible chat-completions client
│   │   └── prompts.py             # Role prompts and JSON schema skeletons
│   ├── models
│   │   └── schemas.py             # Pydantic models for records, stages, outputs, and metrics
│   ├── services
│   │   ├── eval_runner.py         # Asynchronous JSONL evaluation loop and metric aggregation
│   │   └── pipeline_variants.py   # Baselines, ablations, sanitizers, and main pipeline logic
│   └── main.py                    # FastAPI application entry point
├── prep_medxpertqa.py             # Utility script for converting MedXpertQA examples to JSONL
├── requirements.txt
└── .env.example
```

## Installation

Use Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Then set the model endpoint and credentials:

```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini
LLM_USE_JSON_MODE=false
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=700
LLM_TIMEOUT_S=60
LLM_MAX_RETRIES=4
EVAL_CONCURRENCY=4
RUNS_DIR=./runs
DATA_DIR=./data
```

Any OpenAI-compatible chat-completions endpoint can be used, including OpenAI, OpenRouter, and locally hosted compatible gateways. JSON mode is optional and should only be enabled if the selected provider supports `response_format={"type":"json_object"}`.

## Running the service

Start the API server with Uvicorn:

```bash
uvicorn app.main:app --reload
```

The service is mounted under `/v1`.

Health check:

```bash
curl http://localhost:8000/v1/health
```

Expected response:

```json
{
  "ok": true,
  "service": "med-robust-agent",
  "version": "2.1.0"
}
```

## Input format

A question record is represented as JSON with a zero-based answer index:

```json
{
  "id": "example-001",
  "dataset": "medqa_like_eval",
  "question": "A 62-year-old patient presents with ... Which diagnosis is most likely?",
  "choices": [
    "Choice A",
    "Choice B",
    "Choice C",
    "Choice D"
  ],
  "answer_index": 2,
  "meta": {
    "source": "example"
  }
}
```

Dataset evaluation expects one such object per line in JSONL format. The `answer_index` field is used only for evaluation metrics; the pipeline itself receives the question and choices as the reasoning context.

## Single-example inference

Run the full pipeline on one record:

```bash
curl -X POST http://localhost:8000/v1/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "variant": "A0_FULL",
    "record": {
      "id": "case-001",
      "dataset": "demo",
      "question": "A patient vignette goes here.",
      "choices": ["A", "B", "C", "D"],
      "answer_index": 1,
      "meta": {}
    },
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "max_tokens": 700,
    "self_consistency_k": 5,
    "multi_counter_k": 3
  }'
```

The response contains the base prediction, final prediction, final decision, stage-level structured outputs, and aggregate cost information:

```json
{
  "variant": "A0_FULL",
  "base_pred_index": 1,
  "final_pred_index": 1,
  "final_decision": "ANSWERED",
  "stages": {
    "solver": { "...": "..." },
    "counterexample": { "...": "..." },
    "referee": { "...": "..." },
    "repair": null
  },
  "cost": {
    "llm_calls": 3,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "latency_ms": 0
  }
}
```

Token counts depend on whether the selected provider returns usage metadata.

## Dataset evaluation

Launch an asynchronous evaluation job by uploading a JSONL dataset:

```bash
curl -X POST http://localhost:8000/v1/eval/run \
  -F "dataset=@data/eval.jsonl" \
  -F "variant=A0_FULL" \
  -F "model=gpt-4o-mini" \
  -F "temperature=0.0" \
  -F "max_tokens=700"
```

The response returns a `run_id`:

```json
{
  "run_id": "...",
  "state": "PENDING",
  "message": "Queued A0_FULL eval job..."
}
```

Check status and output paths:

```bash
curl http://localhost:8000/v1/eval/{run_id}
```

Completed runs write two artifacts under `RUNS_DIR/{run_id}`:

- `trace_{variant}.jsonl`: one structured trace per example, including stage outputs and cost.
- `metrics_{variant}.json`: aggregate metrics over the dataset and per-dataset subsets.

## Variant sweeps

To compare several experimental conditions in one job:

```bash
curl -X POST http://localhost:8000/v1/eval/sweep \
  -F "dataset=@data/eval.jsonl" \
  -F "variants_csv=B0_DIRECT,B1_COT,B3_SELF_REFINE,A0_FULL,A2_REFEREE_NO_RUBRIC,A3_NO_REPAIR,A5_MULTI_COUNTEREXAMPLES" \
  -F "model=gpt-4o-mini" \
  -F "temperature=0.0"
```

A restricted-attack ablation can be configured with comma-separated attack types:

```bash
curl -X POST http://localhost:8000/v1/eval/run \
  -F "dataset=@data/eval.jsonl" \
  -F "variant=A4_RESTRICTED_ATTACKS" \
  -F "restricted_attack_types_csv=CONTRADICTION,FEATURE_IGNORED"
```

## Metrics

The evaluation runner reports the following quantities:

| Metric | Meaning |
| --- | --- |
| `n` | Number of processed records. |
| `answered` | Number of examples with a final answer. |
| `abstained` | Number of examples where the referee abstained. |
| `base_correct` | Number of examples where the initial solver or baseline prediction was correct. |
| `final_correct` | Number of examples where the final prediction was correct. |
| `fixed` | Number of initially incorrect examples corrected by the pipeline. |
| `broken` | Number of initially correct examples changed to an incorrect answer. |
| `base_acc` | `base_correct / n`. |
| `final_acc` | `final_correct / n`, counting abstentions as not correct. |
| `coverage` | `answered / n`. |
| `selective_acc` | Accuracy over answered examples only. |
| `net_gain` | `(fixed - broken) / n`. |
| `avg_prompt_tokens` | Mean prompt tokens per example. |
| `avg_completion_tokens` | Mean completion tokens per example. |
| `avg_latency_ms` | Mean accumulated LLM latency per example. |
| `avg_llm_calls` | Mean number of LLM calls per example. |

These metrics are also computed per dataset label in the `by_dataset` field.

## Preparing MedXpertQA-style data

The helper script `prep_medxpertqa.py` converts examples from the Hugging Face `TsinghuaC3I/MedXpertQA` dataset into the JSONL schema used by this project. The script removes duplicated answer-choice text from the question stem, converts option dictionaries into ordered choice lists, and maps letter labels to zero-based indices.

Install the optional dataset dependency before using the script:

```bash
pip install datasets
```

Then run:

```bash
python prep_medxpertqa.py
```

The default script writes small development files under `data/`. Adjust `config`, `split`, `n`, and `seed` inside the script for larger experiments.

## Reproducibility notes

For controlled experiments, keep the following settings fixed across variants:

- model identifier and provider;
- temperature and token budget;
- dataset sample and ordering;
- `self_consistency_k` for self-consistency baselines;
- `multi_counter_k` for multi-counterexample experiments;
- evaluation concurrency, if provider rate limits or transient errors are likely to affect results.

The default temperature is `0.0` for deterministic-style runs. `B2_SELF_CONSISTENCY` uses `0.8` when no temperature is supplied, because repeated direct calls are otherwise unlikely to provide meaningful diversity.

Trace files should be retained with metric files. They are necessary for auditing whether gains come from valid corrections, from changes in abstention behavior, or from spurious answer changes.

## Limitations

This framework evaluates model behavior on multiple-choice medical questions; it does not establish clinical validity. Results are sensitive to the underlying model, provider behavior, prompt wording, answer-choice format, and dataset composition

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
