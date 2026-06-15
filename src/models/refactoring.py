# src/models/refactoring.py

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class RefactoringStep(BaseModel):
    operation: str = Field(..., description="ExtractInterface, MoveMethod, SplitClass, IntroducePolymorphism, etc.")
    target_class: str = Field(..., description="FQN of class being modified")
    source_class: Optional[str] = Field(None, description="Source class for move operations")
    affected_methods: List[str] = Field(default_factory=list)
    estimated_loc_impact: int = Field(default=0)
    rationale: str = Field(..., description="Why this step is recommended")

class RefactoringPlan(BaseModel):
    plan_id: str
    title: str
    target_classes: List[str]
    smells_addressed: List[str]
    design_pattern: str
    rationale: str
    steps: List[RefactoringStep]
    estimated_effort: int
    risk_level: str

class ParetoPlanReview(BaseModel):
    reviewed_plan_ids: List[str]
    selected_plan_id: str
    architectural_review_rationale: str
    maintainability_score: float
    domain_safety_score: float
    risk_mitigations: List[str]
    verification_notes: str

class ExecutionPlan(BaseModel):
    batch_id: str
    plan_id: str
    title: str
    target_classes: List[str]
    smells_resolved: List[str]
    resolution_status: str = "VERIFIED_SAFE"
    architectural_review_rationale: str
    refactoring_suggestions: List[Dict[str, Any]]
    metrics_impact: Dict[str, float]
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_model_used: str = "mock-llm"
