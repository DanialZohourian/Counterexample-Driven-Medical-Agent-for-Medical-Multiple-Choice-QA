from __future__ import annotations
import asyncio, json, os
from typing import Any, Dict, List, Optional
import aiofiles
from tqdm import tqdm
from app.core.config import settings
from app.core.util import ensure_dir, dumps_json
from app.core.run_registry import run_registry
from app.models.schemas import QARecord, Variant
from app.services.pipeline_variants import run_variant

def _metrics_init() -> Dict[str, Any]:
    return {"n":0,"answered":0,"base_correct":0,"final_correct":0,"fixed":0,"broken":0,"abstained":0,
            "token_prompt":0,"token_completion":0,"latency_ms":0,"llm_calls":0,"by_dataset":{}}

def _ensure_ds(m: Dict[str, Any], ds: str) -> Dict[str, Any]:
    if ds not in m["by_dataset"]:
        m["by_dataset"][ds] = _metrics_init()
    return m["by_dataset"][ds]

def _correct(pred: Optional[int], gt: int) -> bool:
    return pred is not None and int(pred) == int(gt)

async def _read_jsonl(path: str) -> List[QARecord]:
    out: List[QARecord] = []
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        async for line in f:
            line = line.strip()
            if line:
                out.append(QARecord.model_validate(json.loads(line)))
    return out

async def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    async with aiofiles.open(path, "a", encoding="utf-8") as f:
        await f.write(dumps_json(obj) + "\n")

def _finalize(m: Dict[str, Any]) -> Dict[str, Any]:
    n = max(1, int(m["n"]))
    ans = int(m["answered"])
    out = dict(m)
    out["base_acc"] = m["base_correct"]/n
    out["final_acc"] = m["final_correct"]/n
    out["coverage"] = ans/n
    out["selective_acc"] = (m["final_correct"]/ans) if ans else None
    out["net_gain"] = (m["fixed"]-m["broken"])/n
    out["avg_prompt_tokens"] = m["token_prompt"]/n
    out["avg_completion_tokens"] = m["token_completion"]/n
    out["avg_latency_ms"] = m["latency_ms"]/n
    out["avg_llm_calls"] = m["llm_calls"]/n
    return out

async def run_eval(run_id: str, jsonl_path: str, variant: Variant,
                  model: Optional[str]=None, temperature: Optional[float]=None, max_tokens: Optional[int]=None,
                  self_consistency_k: int=5, multi_counter_k: int=3, restricted_attack_types: Optional[List[str]]=None) -> None:
    ensure_dir(settings.runs_dir)
    run_dir = os.path.join(settings.runs_dir, run_id)
    ensure_dir(run_dir)
    trace_path = os.path.join(run_dir, f"trace_{variant}.jsonl")
    metrics_path = os.path.join(run_dir, f"metrics_{variant}.json")

    recs = await _read_jsonl(jsonl_path)
    run_registry.update(run_id, state="RUNNING", message=f"Running {variant} on {len(recs)} samples...",
                        trace_path=trace_path, metrics_path=metrics_path)

    metrics = _metrics_init()
    sem = asyncio.Semaphore(settings.eval_concurrency)

    async def one(rec: QARecord) -> None:
        async with sem:
            try:
                res = await asyncio.to_thread(run_variant, rec, variant, model, temperature, max_tokens,
                                              self_consistency_k, multi_counter_k, restricted_attack_types)
            except Exception as e:
                await _append_jsonl(trace_path, {"id":rec.id,"dataset":rec.dataset,"error":str(e),"answer_index":rec.answer_index,"variant":variant})
                metrics["n"] += 1; _ensure_ds(metrics, rec.dataset)["n"] += 1
                return

            base_pred = res.base_pred_index
            final_pred = res.final_pred_index
            base_ok = _correct(base_pred, rec.answer_index)
            final_ok = _correct(final_pred, rec.answer_index)

            metrics["n"] += 1
            metrics["token_prompt"] += res.cost.prompt_tokens
            metrics["token_completion"] += res.cost.completion_tokens
            metrics["latency_ms"] += res.cost.latency_ms
            metrics["llm_calls"] += res.cost.llm_calls

            if res.final_decision == "ANSWERED":
                metrics["answered"] += 1
            else:
                metrics["abstained"] += 1

            if base_ok: metrics["base_correct"] += 1
            if final_ok: metrics["final_correct"] += 1
            if (not base_ok) and final_ok: metrics["fixed"] += 1
            if base_ok and (not final_ok): metrics["broken"] += 1

            ds = _ensure_ds(metrics, rec.dataset)
            ds["n"] += 1
            ds["token_prompt"] += res.cost.prompt_tokens
            ds["token_completion"] += res.cost.completion_tokens
            ds["latency_ms"] += res.cost.latency_ms
            ds["llm_calls"] += res.cost.llm_calls
            if res.final_decision == "ANSWERED": ds["answered"] += 1
            else: ds["abstained"] += 1
            if base_ok: ds["base_correct"] += 1
            if final_ok: ds["final_correct"] += 1
            if (not base_ok) and final_ok: ds["fixed"] += 1
            if base_ok and (not final_ok): ds["broken"] += 1

            await _append_jsonl(trace_path, {
                "id": rec.id,
                "dataset": rec.dataset,
                "answer_index": rec.answer_index,
                "base_pred_index": base_pred,
                "final_pred_index": final_pred,
                "final_decision": res.final_decision,
                "variant": variant,
                "stages": res.stages.model_dump(),
                "cost": res.cost.model_dump(),
            })

    tasks = [asyncio.create_task(one(r)) for r in recs]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await fut

    metrics = _finalize(metrics)
    metrics["by_dataset"] = {k:_finalize(v) for k,v in metrics["by_dataset"].items()}
    async with aiofiles.open(metrics_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(metrics, indent=2))

async def run_sweep(run_id: str, jsonl_path: str, variants: List[Variant],
                   model: Optional[str]=None, temperature: Optional[float]=None, max_tokens: Optional[int]=None,
                   self_consistency_k: int=5, multi_counter_k: int=3, restricted_attack_types: Optional[List[str]]=None) -> None:
    run_registry.update(run_id, state="RUNNING", message=f"Sweep started: {variants}")
    try:
        for v in variants:
            run_registry.update(run_id, message=f"Running {v}...")
            await run_eval(run_id, jsonl_path, v, model, temperature, max_tokens, self_consistency_k, multi_counter_k, restricted_attack_types)
        run_registry.update(run_id, state="DONE", message="Sweep complete.")
    except Exception as e:
        run_registry.update(run_id, state="ERROR", message=str(e))
