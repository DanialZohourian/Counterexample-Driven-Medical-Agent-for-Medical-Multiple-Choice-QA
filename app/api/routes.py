from __future__ import annotations

import os
import uuid
from typing import List, Optional

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.core.run_registry import run_registry
from app.core.util import ensure_dir
from app.models.schemas import (
    EvalRunResponse,
    EvalStatusResponse,
    PipelineRequest,
    PipelineResult,
    Variant,
)
from app.services.eval_runner import run_eval, run_sweep
from app.services.pipeline_variants import run_variant

router = APIRouter()

# Keep these in sync with app.models.schemas.Variant / AttackType.
ALLOWED_VARIANTS = {
    "B0_DIRECT",
    "B1_COT",
    "B2_SELF_CONSISTENCY",
    "B3_SELF_REFINE",
    "A0_FULL",
    "A1_SELF_REFINE_ONLY",
    "A2_REFEREE_NO_RUBRIC",
    "A3_NO_REPAIR",
    "A4_RESTRICTED_ATTACKS",
    "A5_MULTI_COUNTEREXAMPLES",
}

ALLOWED_ATTACK_TYPES = {
    "COMPETING_DDX",
    "FEATURE_IGNORED",
    "CONTRADICTION",
    "EINSTELLUNG_TRAP",
    "AMBIGUITY_EXPLOIT",
}


def _parse_csv_list(csv: Optional[str]) -> Optional[List[str]]:
    if csv is None:
        return None
    items = [x.strip() for x in csv.split(",") if x.strip()]
    return items or None


def _validate_variants(variants: List[str]) -> List[str]:
    bad = [v for v in variants if v not in ALLOWED_VARIANTS]
    if bad:
        raise HTTPException(status_code=400, detail=f"Invalid variant(s): {bad}. Allowed: {sorted(ALLOWED_VARIANTS)}")
    return variants


def _validate_attack_types(types_: Optional[List[str]]) -> Optional[List[str]]:
    if not types_:
        return None
    bad = [t for t in types_ if t not in ALLOWED_ATTACK_TYPES]
    if bad:
        raise HTTPException(status_code=400, detail=f"Invalid attack_type(s): {bad}. Allowed: {sorted(ALLOWED_ATTACK_TYPES)}")
    return types_


@router.get("/health")
def health() -> dict:
    return {"ok": True, "service": "med-robust-agent", "version": "2.1.0"}


@router.post("/pipeline", response_model=PipelineResult)
def pipeline(req: PipelineRequest) -> PipelineResult:
    # Convert restricted_attack_types to plain strings for prompts/enforcement.
    rat = [str(x) for x in req.restricted_attack_types] if req.restricted_attack_types else None
    return run_variant(
        req.record,
        req.variant,
        req.model,
        req.temperature,
        req.max_tokens,
        int(req.self_consistency_k),
        int(req.multi_counter_k),
        rat,
    )


@router.post("/eval/run", response_model=EvalRunResponse)
async def eval_run(
    background: BackgroundTasks,
    dataset: UploadFile = File(...),
    # IMPORTANT: use Form() so curl -F "variant=A0_FULL" works with multipart upload.
    variant: Variant = Form("A0_FULL"),
    model: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    max_tokens: Optional[int] = Form(None),
    self_consistency_k: int = Form(5),
    multi_counter_k: int = Form(3),
    restricted_attack_types_csv: Optional[str] = Form(None),
) -> EvalRunResponse:
    ensure_dir(settings.runs_dir)

    # Save upload to disk (streaming, avoids loading whole file into RAM).
    upload_dir = os.path.join(settings.runs_dir, f"upload_{uuid.uuid4()}")
    ensure_dir(upload_dir)
    dataset_path = os.path.join(upload_dir, "dataset.jsonl")

    async with aiofiles.open(dataset_path, "wb") as f:
        while True:
            chunk = await dataset.read(1024 * 1024)
            if not chunk:
                break
            await f.write(chunk)

    run_id = str(uuid.uuid4())
    run_registry.create(run_id)
    run_registry.update(run_id, state="PENDING", message=f"Queued {variant} eval job...")

    rat = _validate_attack_types(_parse_csv_list(restricted_attack_types_csv))

    async def _job() -> None:
        try:
            await run_eval(
                run_id=run_id,
                jsonl_path=dataset_path,
                variant=variant,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                self_consistency_k=self_consistency_k,
                multi_counter_k=multi_counter_k,
                restricted_attack_types=rat,
            )
            run_registry.update(run_id, state="DONE", message="Eval complete.")
        except Exception as e:
            run_registry.update(run_id, state="ERROR", message=str(e))

    background.add_task(_job)

    st = run_registry.get(run_id)
    return EvalRunResponse(run_id=run_id, state=st.state if st else "PENDING", message=st.message if st else "")


@router.post("/eval/sweep", response_model=EvalRunResponse)
async def eval_sweep(
    background: BackgroundTasks,
    dataset: UploadFile = File(...),
    # Provide a comma-separated list for simplicity with multipart upload.
    variants_csv: str = Form("B0_DIRECT,B1_COT,B3_SELF_REFINE,A0_FULL,A2_REFEREE_NO_RUBRIC,A3_NO_REPAIR,A5_MULTI_COUNTEREXAMPLES"),
    model: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    max_tokens: Optional[int] = Form(None),
    self_consistency_k: int = Form(5),
    multi_counter_k: int = Form(3),
    restricted_attack_types_csv: Optional[str] = Form(None),
) -> EvalRunResponse:
    ensure_dir(settings.runs_dir)

    upload_dir = os.path.join(settings.runs_dir, f"upload_{uuid.uuid4()}")
    ensure_dir(upload_dir)
    dataset_path = os.path.join(upload_dir, "dataset.jsonl")

    async with aiofiles.open(dataset_path, "wb") as f:
        while True:
            chunk = await dataset.read(1024 * 1024)
            if not chunk:
                break
            await f.write(chunk)

    variants = _parse_csv_list(variants_csv) or []
    variants = _validate_variants(variants)

    rat = _validate_attack_types(_parse_csv_list(restricted_attack_types_csv))

    run_id = str(uuid.uuid4())
    run_registry.create(run_id)
    run_registry.update(run_id, state="PENDING", message="Queued sweep job...")

    async def _job() -> None:
        await run_sweep(
            run_id=run_id,
            jsonl_path=dataset_path,
            variants=variants,  # runtime is str list; typing is for docs
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            self_consistency_k=self_consistency_k,
            multi_counter_k=multi_counter_k,
            restricted_attack_types=rat,
        )

    background.add_task(_job)

    st = run_registry.get(run_id)
    return EvalRunResponse(run_id=run_id, state=st.state if st else "PENDING", message=st.message if st else "")


@router.get("/eval/{run_id}", response_model=EvalStatusResponse)
def eval_status(run_id: str) -> EvalStatusResponse:
    st = run_registry.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="Run not found.")
    return EvalStatusResponse(
        run_id=st.run_id,
        state=st.state,
        message=st.message,
        metrics_path=st.metrics_path,
        trace_path=st.trace_path,
    )
