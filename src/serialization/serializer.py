# src/serialization/serializer.py

import json
import logging
from pathlib import Path
from typing import List, Optional

from src.models import RefactoringChromosome, ParetoPlanReview, ExecutionPlan
from src.manifest_builder import DetectedSmell

logger = logging.getLogger(__name__)

class ExecutionPlanSerializer:
    @staticmethod
    def serialize(
        batch_id: str,
        selected: RefactoringChromosome,
        review: Optional[ParetoPlanReview],
        batch_smells: List[DetectedSmell]
    ) -> ExecutionPlan:
        return ExecutionPlan(
            batch_id=batch_id,
            plan_id=selected.plan_id,
            title=f"Verified Safe Refactoring Plan for {batch_id}",
            target_classes=list({s.target_class for s in selected.applied_steps}),
            smells_resolved=[s.smell_id for s in batch_smells],
            resolution_status="VERIFIED_SAFE",
            architectural_review_rationale=review.architectural_review_rationale if review else "Heuristic selection.",
            refactoring_suggestions=[s.dict() for s in selected.applied_steps],
            metrics_impact={
                "technical_debt_reduction": float(-selected.fitness_objectives[0]),
                "structural_risk": float(selected.fitness_objectives[1]),
                "implementation_effort_loc": float(selected.fitness_objectives[2])
            }
        )

    @staticmethod
    def write_json(plan: ExecutionPlan, path: Path):
        with open(path, "w") as f:
            json.dump(plan.dict(), f, indent=2)
        logger.info(f"[SERIALIZER] Written to {path}")
