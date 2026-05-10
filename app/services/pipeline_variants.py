from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from app.core.util import dumps_json, is_substring
from app.llm.client import LLMClient
from app.llm import prompts
from app.models.schemas import (
    CounterexampleOutput,
    CounterexampleSet,
    CritiqueOutput,
    LLMCallCost,
    PipelineResult,
    PipelineStages,
    QARecord,
    RefereeOutput,
    RepairOutput,
    SolverOutput,
    Variant,
)

# ---------------- JSON parsing + format-fix retry ----------------

def _parse_json_strict(raw: str) -> Dict[str, Any]:
    return json.loads(raw)


def _call_and_parse(
    client: LLMClient,
    messages: List[Dict[str, str]],
    schema_skeleton: Dict[str, Any],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[Dict[str, Any], int, int, int, int]:
    """Returns (obj, prompt_tokens, completion_tokens, latency_ms, llm_calls_used)."""
    resp = client.chat(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
    pt = resp.usage.prompt_tokens
    ct = resp.usage.completion_tokens
    lat = resp.latency_ms
    calls_used = 1

    try:
        obj = _parse_json_strict(resp.text)
        return obj, pt, ct, lat, calls_used
    except Exception:
        # Best-effort JSON reformatter pass
        fmt_messages = [
            {"role": "system", "content": prompts.FORMATTER_SYSTEM},
            {"role": "user", "content": prompts.build_formatter_user(schema_skeleton, resp.text)},
        ]
        resp2 = client.chat(messages=fmt_messages, model=model, temperature=0.0, max_tokens=max_tokens)
        pt += resp2.usage.prompt_tokens
        ct += resp2.usage.completion_tokens
        lat += resp2.latency_ms
        calls_used += 1
        obj2 = _parse_json_strict(resp2.text)
        return obj2, pt, ct, lat, calls_used


# ---------------- lightweight validators/sanitizers ----------------

def _validate_index(i: int, n: int, name: str) -> None:
    if i < 0 or i >= n:
        raise ValueError(f"{name} out of range: {i} not in [0,{n-1}]")  # pragma: no cover


def _sanitize_solver_quotes(rec: QARecord, solver: SolverOutput) -> SolverOutput:
    """Make quote grounding non-brittle: keep quotes that are substrings, blank the rest."""
    fixed = solver.model_copy(deep=True)
    new_kfs = []
    for kf in fixed.key_features:
        quote = kf.evidence_quote or ""
        if quote and not is_substring(quote, rec.question):
            kf = kf.model_copy(update={"evidence_quote": "-"})
        new_kfs.append(kf)
    fixed.key_features = new_kfs
    return fixed

# ---------------- Robust coercion helpers ----------------

ALLOWED_ATTACK_TYPES = {"COMPETING_DDX","FEATURE_IGNORED","CONTRADICTION","EINSTELLUNG_TRAP","AMBIGUITY_EXPLOIT"}
ALLOWED_DECISIONS = {"KEEP","REVISE","ABSTAIN"}
ALLOWED_WINNERS = {"SOLVER","COUNTEREXAMPLE","NEITHER"}

def _coerce_int(x: Any, default: int = 0) -> int:
    """Coerce ints that may come back as strings or letters (A,B,C...)."""
    if x is None:
        return int(default)
    if isinstance(x, bool):
        return int(default)
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return int(default)
        # letter choice (A,B,C,...) -> index
        if len(s) == 1 and s.isalpha():
            return ord(s.upper()) - ord("A")
        # numeric string
        try:
            return int(float(s))
        except Exception:
            return int(default)
    return int(default)

def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))

def _normalize_attack_type(x: Any) -> str:
    s = str(x or "").strip().upper()
    if s in ALLOWED_ATTACK_TYPES:
        return s
    if s.startswith("COMPETING_"):
        return "COMPETING_DDX"
    if "CONTRAD" in s or "INCONSIST" in s or "CONFLICT" in s:
        return "CONTRADICTION"
    if "EINST" in s or "TRAP" in s or "BIAS" in s:
        return "EINSTELLUNG_TRAP"
    if "IGNORE" in s or "OMIT" in s or "MISSING" in s or "OVERLOOK" in s:
        return "FEATURE_IGNORED"
    if "AMBIG" in s:
        return "AMBIGUITY_EXPLOIT"
    return "AMBIGUITY_EXPLOIT"

def _map_attack_to_subset(norm: str, subset: List[str], attack_argument: str = "") -> str:
    """If we're in A4 restricted mode, map a normalized type into the allowed subset deterministically."""
    subset_u = [str(a).strip().upper() for a in subset]
    subset_u = [a for a in subset_u if a in ALLOWED_ATTACK_TYPES]
    if not subset_u:
        return norm
    if norm in subset_u:
        return norm
    arg = (attack_argument or "").lower()
    # heuristic mapping
    if any(k in arg for k in ["contrad", "inconsist", "conflict"]):
        if "CONTRADICTION" in subset_u:
            return "CONTRADICTION"
    if any(k in arg for k in ["ignore", "omit", "overlook", "missing"]):
        if "FEATURE_IGNORED" in subset_u:
            return "FEATURE_IGNORED"
    # fall back to first allowed to keep pipeline running
    return subset_u[0]

def _ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}

def _ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []

def _sanitize_counter_obj(obj: Dict[str, Any], n_choices: int, solver_pred: int, allowed_attack_types: Optional[List[str]]) -> Dict[str, Any]:
    o = dict(obj or {})
    # alt_index coercion
    o["alt_index"] = _coerce_int(o.get("alt_index"), default=0)
    if int(o["alt_index"]) == int(solver_pred):
        # pick a different index deterministically
        o["alt_index"] = 0 if solver_pred != 0 else (1 if n_choices > 1 else 0)
    o["alt_index"] = _clamp(int(o["alt_index"]), 0, max(0, n_choices - 1))

    # attack_type normalization + restriction mapping
    attack_argument = str(o.get("attack_argument") or "")
    norm = _normalize_attack_type(o.get("attack_type"))
    if allowed_attack_types is not None:
        norm = _map_attack_to_subset(norm, allowed_attack_types, attack_argument)
    o["attack_type"] = norm

    # nested structures
    supp = _ensure_dict(o.get("attack_support"))
    quotes = _ensure_list(supp.get("question_quotes"))
    supp["question_quotes"] = [str(q) for q in quotes]
    supp["feature_targeted"] = str(supp.get("feature_targeted") or "")
    o["attack_support"] = supp

    mp = _ensure_dict(o.get("minimal_perturbation"))
    mp["edit"] = str(mp.get("edit") or "")
    mp["effect"] = str(mp.get("effect") or "")
    o["minimal_perturbation"] = mp

    o["attack_argument"] = attack_argument
    return o

def _sanitize_referee_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    o = dict(obj or {})
    # normalize decision/winner
    dec = str(o.get("decision") or "ABSTAIN").strip().upper()
    win = str(o.get("winner") or "NEITHER").strip().upper()
    if dec not in ALLOWED_DECISIONS:
        dec = "ABSTAIN"
    if win not in ALLOWED_WINNERS:
        win = "NEITHER"

    rub = _ensure_dict(o.get("rubric_scores"))
    rub_fixed = {
        "fact_consistency": _clamp(_coerce_int(rub.get("fact_consistency"), 0), 0, 3),
        "coverage_of_findings": _clamp(_coerce_int(rub.get("coverage_of_findings"), 0), 0, 3),
        "clinical_plausibility": _clamp(_coerce_int(rub.get("clinical_plausibility"), 0), 0, 3),
        "parsimony": _clamp(_coerce_int(rub.get("parsimony"), 0), 0, 3),
    }

    action = _ensure_dict(o.get("action"))
    action_fixed = {
        "required_fix": str(action.get("required_fix") or ""),
        "abstain_reason": str(action.get("abstain_reason") or ""),
    }

    # Policy consistency auto-fix (prevents pipeline crashes)
    if win == "COUNTEREXAMPLE" and dec == "KEEP":
        dec = "REVISE"
    if dec == "REVISE" and win != "COUNTEREXAMPLE":
        win = "COUNTEREXAMPLE"
    if dec == "ABSTAIN":
        win = "NEITHER"

    o["decision"] = dec
    o["winner"] = win
    o["rubric_scores"] = rub_fixed
    o["rationale_short"] = str(o.get("rationale_short") or "")
    o["action"] = action_fixed
    return o




def _sanitize_counter_quotes(rec: QARecord, counter: CounterexampleOutput) -> CounterexampleOutput:
    fixed = counter.model_copy(deep=True)
    q = []
    for s in fixed.attack_support.question_quotes:
        s2 = (s or "").strip()
        if s2 and is_substring(s2, rec.question):
            q.append(s2)
    fixed.attack_support = fixed.attack_support.model_copy(update={"question_quotes": q})
    return fixed


# ---------------- Stage runners ----------------

def _run_direct(
    client: LLMClient,
    rec: QARecord,
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[int, int, int, int, int]:
    messages = [
        {"role": "system", "content": prompts.BASE_SYSTEM + "\n" + prompts.ROLE_DIRECT},
        {"role": "user", "content": prompts.build_direct_user(rec.id, rec.dataset, rec.question, rec.choices)},
    ]
    obj, pt, ct, lat, calls = _call_and_parse(client, messages, prompts.DIRECT_SCHEMA, model, temperature, max_tokens)
    # robust coercion
    obj["pred_index"] = _clamp(_coerce_int(obj.get("pred_index"), 0), 0, max(0, len(rec.choices) - 1))
    try:
        obj["confidence"] = float(obj.get("confidence", 0.0))
    except Exception:
        obj["confidence"] = 0.0
    obj["confidence"] = max(0.0, min(1.0, float(obj["confidence"])))
    pred = int(obj["pred_index"])
    _validate_index(pred, len(rec.choices), "direct.pred_index")
    return pred, pt, ct, lat, calls


def _run_solver(
    client: LLMClient,
    rec: QARecord,
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[SolverOutput, Dict[str, Any], int, int, int, int]:
    messages = [
        {"role": "system", "content": prompts.BASE_SYSTEM + "\n" + prompts.ROLE_SOLVER_COT},
        {"role": "user", "content": prompts.build_solver_user(rec.id, rec.dataset, rec.question, rec.choices)},
    ]
    obj, pt, ct, lat, calls = _call_and_parse(client, messages, prompts.SOLVER_SCHEMA, model, temperature, max_tokens)
    # robust coercion
    obj["pred_index"] = _clamp(_coerce_int(obj.get("pred_index"), 0), 0, max(0, len(rec.choices) - 1))
    try:
        obj["confidence"] = float(obj.get("confidence", 0.0))
    except Exception:
        obj["confidence"] = 0.0
    obj["confidence"] = max(0.0, min(1.0, float(obj["confidence"])))
    solver = SolverOutput.model_validate(obj)
    _validate_index(int(solver.pred_index), len(rec.choices), "solver.pred_index")
    solver = _sanitize_solver_quotes(rec, solver)
    return solver, solver.model_dump(), pt, ct, lat, calls


def _run_counterexample(
    client: LLMClient,
    rec: QARecord,
    solver_obj: Dict[str, Any],
    solver_pred: int,
    allowed_attack_types: Optional[List[str]],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[CounterexampleOutput, Dict[str, Any], int, int, int, int]:
    messages = [
        {"role": "system", "content": prompts.BASE_SYSTEM + "\n" + prompts.ROLE_COUNTER},
        {
            "role": "user",
            "content": prompts.build_counter_user(
                rec.id,
                rec.question,
                rec.choices,
                dumps_json(solver_obj),
                allowed_attack_types,
            ),
        },
    ]
    obj, pt, ct, lat, calls = _call_and_parse(client, messages, prompts.COUNTER_SCHEMA, model, temperature, max_tokens)
    obj = _sanitize_counter_obj(obj, len(rec.choices), solver_pred, allowed_attack_types)
    counter = CounterexampleOutput.model_validate(obj)
    _validate_index(int(counter.alt_index), len(rec.choices), "counter.alt_index")
    if int(counter.alt_index) == int(solver_pred):
        raise ValueError("Counterexample alt_index equals solver pred_index")
    counter = _sanitize_counter_quotes(rec, counter)

    # Enforce attack-type restriction if provided (A4 ablation).
    if allowed_attack_types is not None:
        allowed_u = [str(a).strip().upper() for a in allowed_attack_types]
        if counter.attack_type not in allowed_u:
            # should be rare after sanitization; keep running
            counter = counter.model_copy(update={"attack_type": allowed_u[0] if allowed_u else "AMBIGUITY_EXPLOIT"})

    return counter, counter.model_dump(), pt, ct, lat, calls


def _run_counterexample_set(
    client: LLMClient,
    rec: QARecord,
    solver_obj: Dict[str, Any],
    solver_pred: int,
    k: int,
    allowed_attack_types: Optional[List[str]],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[CounterexampleSet, Dict[str, Any], int, int, int, int]:
    messages = [
        {"role": "system", "content": prompts.BASE_SYSTEM + "\n" + prompts.ROLE_COUNTER},
        {
            "role": "user",
            "content": prompts.build_counter_set_user(
                rec.id,
                rec.question,
                rec.choices,
                dumps_json(solver_obj),
                int(k),
                allowed_attack_types,
            ),
        },
    ]
    obj, pt, ct, lat, calls = _call_and_parse(client, messages, prompts.COUNTER_SET_SCHEMA, model, temperature, max_tokens)
    # sanitize each item before validation
    if isinstance(obj, dict) and "items" in obj and isinstance(obj.get("items"), list):
        obj["items"] = [_sanitize_counter_obj(it if isinstance(it, dict) else {}, len(rec.choices), solver_pred, allowed_attack_types) for it in obj["items"]]
    cset = CounterexampleSet.model_validate(obj)

    # basic validation + sanitization
    fixed_items: List[CounterexampleOutput] = []
    for item in cset.items:
        _validate_index(int(item.alt_index), len(rec.choices), "counter_set.alt_index")
        if int(item.alt_index) == int(solver_pred):
            raise ValueError("CounterexampleSet item.alt_index equals solver pred_index")
        item = _sanitize_counter_quotes(rec, item)
        if allowed_attack_types is not None and item.attack_type not in allowed_attack_types:
            raise ValueError(f"attack_type '{item.attack_type}' not in allowed list {allowed_attack_types}")
        fixed_items.append(item)

    cset = cset.model_copy(update={"items": fixed_items})
    return cset, cset.model_dump(), pt, ct, lat, calls


def _run_referee(
    client: LLMClient,
    rec: QARecord,
    solver_obj: Dict[str, Any],
    counter_obj: Dict[str, Any],
    free_referee: bool,
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[RefereeOutput, Dict[str, Any], int, int, int, int]:
    role = prompts.ROLE_REFEREE_FREE if free_referee else prompts.ROLE_REFEREE_RUBRIC
    messages = [
        {"role": "system", "content": prompts.BASE_SYSTEM + "\n" + role},
        {
            "role": "user",
            "content": prompts.build_referee_user(
                rec.id,
                rec.question,
                rec.choices,
                dumps_json(solver_obj),
                dumps_json(counter_obj),
            ),
        },
    ]
    obj, pt, ct, lat, calls = _call_and_parse(client, messages, prompts.REFEREE_SCHEMA, model, temperature, max_tokens)
    obj = _sanitize_referee_obj(obj)
    ref = RefereeOutput.model_validate(obj)
    return ref, ref.model_dump(), pt, ct, lat, calls


def _run_referee_on_set(
    client: LLMClient,
    rec: QARecord,
    solver_obj: Dict[str, Any],
    counter_set_obj: Dict[str, Any],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[RefereeOutput, Dict[str, Any], int, int, int, int]:
    messages = [
        {"role": "system", "content": prompts.BASE_SYSTEM + "\n" + prompts.ROLE_REFEREE_RUBRIC},
        {
            "role": "user",
            "content": prompts.build_referee_set_user(
                rec.id,
                rec.question,
                rec.choices,
                dumps_json(solver_obj),
                dumps_json(counter_set_obj),
            ),
        },
    ]
    obj, pt, ct, lat, calls = _call_and_parse(client, messages, prompts.REFEREE_SCHEMA, model, temperature, max_tokens)
    obj = _sanitize_referee_obj(obj)
    ref = RefereeOutput.model_validate(obj)
    return ref, ref.model_dump(), pt, ct, lat, calls


def _run_selector_on_set(
    client: LLMClient,
    rec: QARecord,
    solver_obj: Dict[str, Any],
    counter_set_obj: Dict[str, Any],
    referee_obj: Dict[str, Any],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[int, int, int, int, int]:
    """Choose which counterexample item best matches the referee's reasoning/fix."""
    messages = [
        {"role": "system", "content": prompts.BASE_SYSTEM + "\n" + prompts.ROLE_SELECTOR},
        {
            "role": "user",
            "content": prompts.build_selector_user(
                rec.id,
                rec.question,
                rec.choices,
                dumps_json(solver_obj),
                dumps_json(counter_set_obj),
                dumps_json(referee_obj),
            ),
        },
    ]
    obj, pt, ct, lat, calls = _call_and_parse(client, messages, prompts.SELECT_SCHEMA, model, temperature, max_tokens)
    sel = int(obj.get("selected_index", 0))
    # bounds check against actual set length
    items = counter_set_obj.get("items") or []
    if not isinstance(items, list) or not items:
        return 0, pt, ct, lat, calls
    if sel < 0:
        sel = 0
    if sel >= len(items):
        sel = len(items) - 1
    return sel, pt, ct, lat, calls


def _run_repair(
    client: LLMClient,
    rec: QARecord,
    solver_obj: Dict[str, Any],
    counter_obj: Dict[str, Any],
    referee_obj: Dict[str, Any],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[RepairOutput, Dict[str, Any], int, int, int, int]:
    messages = [
        {"role": "system", "content": prompts.BASE_SYSTEM + "\n" + prompts.ROLE_REPAIR},
        {
            "role": "user",
            "content": prompts.build_repair_user(
                rec.id,
                rec.question,
                rec.choices,
                dumps_json(solver_obj),
                dumps_json(counter_obj),
                dumps_json(referee_obj),
            ),
        },
    ]
    obj, pt, ct, lat, calls = _call_and_parse(client, messages, prompts.REPAIR_SCHEMA, model, temperature, max_tokens)
    # robust coercion
    obj["pred_index"] = _clamp(_coerce_int(obj.get("pred_index"), 0), 0, max(0, len(rec.choices) - 1))
    try:
        obj["confidence"] = float(obj.get("confidence", 0.0))
    except Exception:
        obj["confidence"] = 0.0
    obj["confidence"] = max(0.0, min(1.0, float(obj["confidence"])))
    rep = RepairOutput.model_validate(obj)
    _validate_index(int(rep.pred_index), len(rec.choices), "repair.pred_index")
    return rep, rep.model_dump(), pt, ct, lat, calls


def _run_critique(
    client: LLMClient,
    rec: QARecord,
    solver_obj: Dict[str, Any],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[CritiqueOutput, Dict[str, Any], int, int, int, int]:
    messages = [
        {"role": "system", "content": prompts.BASE_SYSTEM + "\n" + prompts.ROLE_CRITIQUE},
        {
            "role": "user",
            "content": prompts.build_critique_user(rec.id, rec.question, rec.choices, dumps_json(solver_obj)),
        },
    ]
    obj, pt, ct, lat, calls = _call_and_parse(client, messages, prompts.CRITIQUE_SCHEMA, model, temperature, max_tokens)
    crit = CritiqueOutput.model_validate(obj)
    if crit.suggested_alternative_index is not None:
        _validate_index(int(crit.suggested_alternative_index), len(rec.choices), "critique.suggested_alternative_index")
    return crit, crit.model_dump(), pt, ct, lat, calls


# ---------------- Variants ----------------

DEFAULT_RESTRICTED_ATTACKS = ["CONTRADICTION", "FEATURE_IGNORED"]


def run_variant(
    rec: QARecord,
    variant: Variant,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    self_consistency_k: int = 5,
    multi_counter_k: int = 3,
    restricted_attack_types: Optional[List[str]] = None,
) -> PipelineResult:
    client = LLMClient()
    stages = PipelineStages()

    llm_calls = 0
    pt = 0
    ct = 0
    lat = 0

    def _add_cost(p: int, c: int, l: int, calls: int) -> None:
        nonlocal pt, ct, lat, llm_calls
        pt += int(p)
        ct += int(c)
        lat += int(l)
        llm_calls += int(calls)

    # ---------- Baselines ----------
    if variant == "B0_DIRECT":
        pred, p, c, l, calls = _run_direct(client, rec, model, temperature, max_tokens)
        _add_cost(p, c, l, calls)
        return PipelineResult(
            variant=variant,
            base_pred_index=pred,
            final_pred_index=pred,
            final_decision="ANSWERED",
            stages=stages,
            cost=LLMCallCost(llm_calls=llm_calls, prompt_tokens=pt, completion_tokens=ct, latency_ms=lat),
        )

    if variant == "B1_COT":
        solver, solver_obj, p, c, l, calls = _run_solver(client, rec, model, temperature, max_tokens)
        stages.solver = solver
        _add_cost(p, c, l, calls)
        pred = int(solver.pred_index)
        return PipelineResult(
            variant=variant,
            base_pred_index=pred,
            final_pred_index=pred,
            final_decision="ANSWERED",
            stages=stages,
            cost=LLMCallCost(llm_calls=llm_calls, prompt_tokens=pt, completion_tokens=ct, latency_ms=lat),
        )

    if variant == "B2_SELF_CONSISTENCY":
        k = int(self_consistency_k)
        votes: Dict[int, int] = {}
        for _ in range(k):
            pred, p, c, l, calls = _run_direct(
                client,
                rec,
                model,
                temperature=(0.8 if temperature is None else float(temperature)),
                max_tokens=max_tokens,
            )
            _add_cost(p, c, l, calls)
            votes[pred] = votes.get(pred, 0) + 1
        best = sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]
        return PipelineResult(
            variant=variant,
            base_pred_index=best,
            final_pred_index=best,
            final_decision="ANSWERED",
            stages=stages,
            cost=LLMCallCost(llm_calls=llm_calls, prompt_tokens=pt, completion_tokens=ct, latency_ms=lat),
        )

    if variant in ("B3_SELF_REFINE", "A1_SELF_REFINE_ONLY"):
        solver, solver_obj, p1, c1, l1, calls1 = _run_solver(client, rec, model, temperature, max_tokens)
        stages.solver = solver
        _add_cost(p1, c1, l1, calls1)

        crit, crit_obj, p2, c2, l2, calls2 = _run_critique(client, rec, solver_obj, model, temperature, max_tokens)
        stages.critique = crit
        _add_cost(p2, c2, l2, calls2)

        if crit.suggested_alternative_index is None:
            pred = int(solver.pred_index)
            return PipelineResult(
                variant=variant,
                base_pred_index=pred,
                final_pred_index=pred,
                final_decision="ANSWERED",
                stages=stages,
                cost=LLMCallCost(llm_calls=llm_calls, prompt_tokens=pt, completion_tokens=ct, latency_ms=lat),
            )

        # Build a minimal counter/referee scaffolding so we can reuse the Repair agent.
        pseudo_counter = {
            "alt_index": int(crit.suggested_alternative_index),
            "attack_type": "AMBIGUITY_EXPLOIT",
            "attack_argument": "Self-critique suggested an alternative.",
            "attack_support": {"question_quotes": [rec.question[:80]], "feature_targeted": "critique"},
            "minimal_perturbation": {"edit": "", "effect": ""},
        }
        pseudo_ref = {
            "decision": "REVISE",
            "winner": "COUNTEREXAMPLE",
            "rubric_scores": {"fact_consistency": 0, "coverage_of_findings": 0, "clinical_plausibility": 0, "parsimony": 0},
            "rationale_short": "Self-refine revision.",
            "action": {"required_fix": crit.required_fix, "abstain_reason": ""},
        }

        rep, rep_obj, p3, c3, l3, calls3 = _run_repair(client, rec, solver_obj, pseudo_counter, pseudo_ref, model, temperature, max_tokens)
        stages.repair = rep
        _add_cost(p3, c3, l3, calls3)

        return PipelineResult(
            variant=variant,
            base_pred_index=int(solver.pred_index),
            final_pred_index=int(rep.pred_index),
            final_decision="ANSWERED",
            stages=stages,
            cost=LLMCallCost(llm_calls=llm_calls, prompt_tokens=pt, completion_tokens=ct, latency_ms=lat),
        )

    # ---------- Main pipelines ----------
    solver, solver_obj, p1, c1, l1, calls1 = _run_solver(client, rec, model, temperature, max_tokens)
    stages.solver = solver
    _add_cost(p1, c1, l1, calls1)
    base_pred = int(solver.pred_index)

    if variant == "A5_MULTI_COUNTEREXAMPLES":
        # You can still pass restricted_attack_types to limit the counterexample generator.
        cset, cset_obj, p2, c2, l2, calls2 = _run_counterexample_set(
            client,
            rec,
            solver_obj,
            base_pred,
            int(multi_counter_k),
            restricted_attack_types,
            model,
            temperature,
            max_tokens,
        )
        stages.counterexample_set = cset
        _add_cost(p2, c2, l2, calls2)

        ref, ref_obj, p3, c3, l3, calls3 = _run_referee_on_set(client, rec, solver_obj, cset_obj, model, temperature, max_tokens)
        stages.referee = ref
        _add_cost(p3, c3, l3, calls3)

        if ref.decision == "KEEP":
            return PipelineResult(
                variant=variant,
                base_pred_index=base_pred,
                final_pred_index=base_pred,
                final_decision="ANSWERED",
                stages=stages,
                cost=LLMCallCost(llm_calls=llm_calls, prompt_tokens=pt, completion_tokens=ct, latency_ms=lat),
            )

        if ref.decision == "ABSTAIN":
            return PipelineResult(
                variant=variant,
                base_pred_index=base_pred,
                final_pred_index=None,
                final_decision="ABSTAINED",
                stages=stages,
                cost=LLMCallCost(llm_calls=llm_calls, prompt_tokens=pt, completion_tokens=ct, latency_ms=lat),
            )

        # REVISE: choose a counterexample consistent with the referee output.
        sel_idx, p_sel, c_sel, l_sel, calls_sel = _run_selector_on_set(
            client, rec, solver_obj, cset_obj, ref_obj, model, temperature, max_tokens
        )
        _add_cost(p_sel, c_sel, l_sel, calls_sel)

        chosen = cset.items[int(sel_idx)]
        stages.counterexample = chosen
        counter_obj = chosen.model_dump()

        rep, rep_obj, p4, c4, l4, calls4 = _run_repair(client, rec, solver_obj, counter_obj, ref_obj, model, temperature, max_tokens)
        stages.repair = rep
        _add_cost(p4, c4, l4, calls4)

        return PipelineResult(
            variant=variant,
            base_pred_index=base_pred,
            final_pred_index=int(rep.pred_index),
            final_decision="ANSWERED",
            stages=stages,
            cost=LLMCallCost(llm_calls=llm_calls, prompt_tokens=pt, completion_tokens=ct, latency_ms=lat),
        )

    # A4 should always restrict; fall back to sensible defaults if omitted.
    allowed = None
    if variant == "A4_RESTRICTED_ATTACKS":
        allowed = restricted_attack_types or DEFAULT_RESTRICTED_ATTACKS

    counter, counter_obj, p2, c2, l2, calls2 = _run_counterexample(
        client, rec, solver_obj, base_pred, allowed, model, temperature, max_tokens
    )
    stages.counterexample = counter
    _add_cost(p2, c2, l2, calls2)

    ref, ref_obj, p3, c3, l3, calls3 = _run_referee(
        client, rec, solver_obj, counter_obj, free_referee=(variant == "A2_REFEREE_NO_RUBRIC"), model=model, temperature=temperature, max_tokens=max_tokens
    )
    stages.referee = ref
    _add_cost(p3, c3, l3, calls3)

    # Consistency guards
    if ref.winner == "COUNTEREXAMPLE" and ref.decision == "KEEP":
        raise RuntimeError("Invalid referee: COUNTEREXAMPLE winner but KEEP")
    if ref.decision == "REVISE" and ref.winner != "COUNTEREXAMPLE":
        raise RuntimeError("Invalid referee: REVISE but winner != COUNTEREXAMPLE")

    if ref.decision == "KEEP":
        return PipelineResult(
            variant=variant,
            base_pred_index=base_pred,
            final_pred_index=base_pred,
            final_decision="ANSWERED",
            stages=stages,
            cost=LLMCallCost(llm_calls=llm_calls, prompt_tokens=pt, completion_tokens=ct, latency_ms=lat),
        )

    if ref.decision == "ABSTAIN":
        return PipelineResult(
            variant=variant,
            base_pred_index=base_pred,
            final_pred_index=None,
            final_decision="ABSTAINED",
            stages=stages,
            cost=LLMCallCost(llm_calls=llm_calls, prompt_tokens=pt, completion_tokens=ct, latency_ms=lat),
        )

    # REVISE
    if variant == "A3_NO_REPAIR":
        return PipelineResult(
            variant=variant,
            base_pred_index=base_pred,
            final_pred_index=int(counter.alt_index),
            final_decision="ANSWERED",
            stages=stages,
            cost=LLMCallCost(llm_calls=llm_calls, prompt_tokens=pt, completion_tokens=ct, latency_ms=lat),
        )

    rep, rep_obj, p4, c4, l4, calls4 = _run_repair(client, rec, solver_obj, counter_obj, ref_obj, model, temperature, max_tokens)
    stages.repair = rep
    _add_cost(p4, c4, l4, calls4)

    return PipelineResult(
        variant=variant,
        base_pred_index=base_pred,
        final_pred_index=int(rep.pred_index),
        final_decision="ANSWERED",
        stages=stages,
        cost=LLMCallCost(llm_calls=llm_calls, prompt_tokens=pt, completion_tokens=ct, latency_ms=lat),
    )
