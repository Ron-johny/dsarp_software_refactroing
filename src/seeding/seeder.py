# src/seeding/seeder.py

import json
import logging
import networkx as nx
from typing import List
from pydantic import ValidationError

from src.models import RefactoringPlan, RefactoringStep
from src.manifest_builder import DetectedSmell
from src.utils.mock_llm import generate_seed_plans
from src.llm import LLMClient

logger = logging.getLogger(__name__)

class LLMPopulationSeeder:
    def seed_population(
        self,
        batch_id: str,
        target_classes: List[str],
        batch_smells: List[DetectedSmell],
        isolated_graph: nx.DiGraph,
        population_size: int = 20
    ) -> List[RefactoringPlan]:
        logger.info(f"[SEEDER] Generating seed plans for {batch_id}")

        # response_json = generate_seed_plans(batch_id, target_classes, batch_smells)

        client = LLMClient()

        prompt = self._build_prompt(batch_id, target_classes, batch_smells)
        raw = client.generate_json(prompt)

        try:
            # raw = json.loads(response_json)
            if not isinstance(raw, list):
                raw = [raw]
            plans = []
            for item in raw:
                try:
                    plans.append(RefactoringPlan(**item))
                except ValidationError as e:
                    logger.warning(f"[SEEDER] Skipping invalid plan: {e}")
            logger.info(f"[SEEDER] Parsed {len(plans)} seed plans")
            return plans
        except json.JSONDecodeError as e:
            logger.error(f"[SEEDER] JSON parse error: {e}")
            return self._fallback_plans(batch_id, target_classes, batch_smells)

    def _fallback_plans(self, batch_id: str, target_classes: List[str], batch_smells: List[DetectedSmell]) -> List[RefactoringPlan]:
        return [RefactoringPlan(
            plan_id=f"FALLBACK_{batch_id}",
            title="Heuristic Fallback",
            target_classes=target_classes,
            smells_addressed=[s.smell_id for s in batch_smells],
            design_pattern="GenericRefactor",
            rationale="Fallback when LLM unavailable",
            steps=[RefactoringStep(
                operation="ExtractInterface",
                target_class=target_classes[0] if target_classes else "Unknown",
                estimated_loc_impact=100,
                rationale="Generic refactoring"
            )],
            estimated_effort=15,
            risk_level="MEDIUM"
        )]
    
    def _build_prompt(self, batch_id, target_classes, batch_smells):
        return f"""
            You are a software architecture refactoring expert.

            Generate 5 candidate refactoring plans.

            Output JSON ONLY.

            Schema:

            [
            {{
                "plan_id": "...",
                "title": "...",
                "target_classes": [],
                "smells_addressed": [],
                "design_pattern": "...",
                "rationale": "...",
                "steps": [
                {{
                    "operation": "...",
                    "target_class": "...",
                    "source_class": null,
                    "affected_methods": [],
                    "estimated_loc_impact": 100,
                    "rationale": "..."
                }}
                ],
                "estimated_effort": 10,
                "risk_level": "LOW"
            }}
            ]

            Batch:

            {batch_id}

            Target Classes:

            {target_classes}

            Smells:

            {[s.smell_type for s in batch_smells]}
        """
