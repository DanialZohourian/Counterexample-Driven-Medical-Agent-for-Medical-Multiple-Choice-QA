from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, conint, confloat

class QARecord(BaseModel):
    id: str
    dataset: str
    question: str
    choices: List[str]
    answer_index: conint(ge=0)
    meta: Dict[str, Any] = Field(default_factory=dict)

Variant = Literal[
    "B0_DIRECT","B1_COT","B2_SELF_CONSISTENCY","B3_SELF_REFINE",
    "A0_FULL","A1_SELF_REFINE_ONLY","A2_REFEREE_NO_RUBRIC","A3_NO_REPAIR","A4_RESTRICTED_ATTACKS","A5_MULTI_COUNTEREXAMPLES"
]

class KeyFeature(BaseModel):
    name: str
    value: str
    evidence_quote: str

class AltChoice(BaseModel):
    index: conint(ge=0)
    why_not: str

class SolverDecisionTrace(BaseModel):
    top_alternatives: List[AltChoice]

class SolverOutput(BaseModel):
    pred_index: conint(ge=0)
    confidence: confloat(ge=0.0, le=1.0)
    key_features: List[KeyFeature]
    rationale_short: str
    decision_trace: SolverDecisionTrace

AttackType = Literal["COMPETING_DDX","FEATURE_IGNORED","CONTRADICTION","EINSTELLUNG_TRAP","AMBIGUITY_EXPLOIT"]

class MinimalPerturbation(BaseModel):
    edit: str
    effect: str

class AttackSupport(BaseModel):
    question_quotes: List[str]
    feature_targeted: str

class CounterexampleOutput(BaseModel):
    alt_index: conint(ge=0)
    attack_type: AttackType
    attack_argument: str
    attack_support: AttackSupport
    minimal_perturbation: MinimalPerturbation

class CounterexampleSet(BaseModel):
    items: List[CounterexampleOutput]

Decision = Literal["KEEP","REVISE","ABSTAIN"]
Winner = Literal["SOLVER","COUNTEREXAMPLE","NEITHER"]

class RubricScores(BaseModel):
    fact_consistency: conint(ge=0, le=3)
    coverage_of_findings: conint(ge=0, le=3)
    clinical_plausibility: conint(ge=0, le=3)
    parsimony: conint(ge=0, le=3)

class RefereeAction(BaseModel):
    required_fix: str = ""
    abstain_reason: str = ""

class RefereeOutput(BaseModel):
    decision: Decision
    winner: Winner
    rubric_scores: RubricScores
    rationale_short: str
    action: RefereeAction

class CritiqueOutput(BaseModel):
    issues: List[str]
    required_fix: str
    suggested_alternative_index: Optional[int] = None

class RepairOutput(BaseModel):
    pred_index: conint(ge=0)
    confidence: confloat(ge=0.0, le=1.0)
    rationale_short: str
    counterexample_addressed: str

class PipelineRequest(BaseModel):
    record: QARecord
    variant: Variant = "A0_FULL"
    self_consistency_k: conint(ge=1, le=15) = 5
    multi_counter_k: conint(ge=1, le=5) = 3
    restricted_attack_types: Optional[List[AttackType]] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class LLMCallCost(BaseModel):
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int

class PipelineStages(BaseModel):
    solver: Optional[SolverOutput] = None
    counterexample: Optional[CounterexampleOutput] = None
    counterexample_set: Optional[CounterexampleSet] = None
    referee: Optional[RefereeOutput] = None
    critique: Optional[CritiqueOutput] = None
    repair: Optional[RepairOutput] = None

class PipelineResult(BaseModel):
    variant: Variant
    base_pred_index: Optional[int] = None
    final_pred_index: Optional[int] = None
    final_decision: Literal["ANSWERED","ABSTAINED"]
    stages: PipelineStages
    cost: LLMCallCost

class EvalRunResponse(BaseModel):
    run_id: str
    state: str
    message: str

class EvalStatusResponse(BaseModel):
    run_id: str
    state: str
    message: str
    metrics_path: Optional[str] = None
    trace_path: Optional[str] = None
