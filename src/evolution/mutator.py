# src/evolution/mutator.py

import json
import logging
import random
import networkx as nx
from typing import Dict, List
from pydantic import ValidationError

from src.models import RefactoringChromosome, RefactoringStep
from src.manifest_builder import DetectedSmell
from src.evolution.gate import AntiRegressionGate
from src.utils.graph_utils import get_structural_neighbors
from src.utils.mock_llm import generate_mutation_steps
from src.llm import LLMClient


logger = logging.getLogger(__name__)

class LLMSemanticMutator:
    def __init__(
        self,
        isolated_graph: nx.DiGraph,
        batch_smells: List[DetectedSmell],
        anti_regression_gate: AntiRegressionGate
    ):
        self.isolated_graph = isolated_graph
        self.batch_smells = batch_smells
        self.smell_by_id = {s.smell_id: s for s in batch_smells}
        self.gate = anti_regression_gate
        self.mutation_count = 0
        self.accepted_mutations = 0

    def mutate(self, chromosome: RefactoringChromosome) -> RefactoringChromosome:
        self.mutation_count += 1

        if not chromosome.smell_sequence:
            return chromosome

        mutant = chromosome.copy()

        # Pick a random smell from the sequence to target
        target_smell_id = random.choice(mutant.smell_sequence)
        if target_smell_id not in self.smell_by_id:
            return mutant

        smell = self.smell_by_id[target_smell_id]
        target_class = smell.target_fqn

        neighbors = get_structural_neighbors(self.isolated_graph, target_class)

        logger.info(f"[MUTATOR] Semantic mutation for smell '{smell.smell_type}' on '{target_class}'")

        # Get LLM-guided steps
        steps = self._get_mutation_steps(smell, target_class, neighbors)
        mutant.applied_steps.extend(steps)

        # Veto gate check
        if self.gate.validate_mutation(mutant):
            self.accepted_mutations += 1
            logger.info(f"[MUTATOR] Mutation accepted (+{len(steps)} steps)")
        else:
            logger.warning(f"[MUTATOR] Mutation rejected by veto gate")

        return mutant

    def _get_mutation_steps(self, smell, target_class, neighbors):
        # response_json = generate_mutation_steps(smell_type, target_class)
        client = LLMClient()
        mutation_prompt = self._mutation_prompt(smell, target_class, neighbors)
        raw = client.generate_json(
            mutation_prompt
        )
        try:
            # raw = json.loads(response_json)
            if not isinstance(raw, list):
                raw = [raw]
            steps = []
            for item in raw:
                try:
                    steps.append(RefactoringStep(**item))
                except ValidationError as e:
                    logger.warning(f"[MUTATOR] Invalid step skipped: {e}")
            return steps
        except json.JSONDecodeError:
            return []
        
    def _mutation_prompt(self, smell, target_class, neighbors):
        return f"""
            You are an architecture refactoring expert.

            Smell Type:
            {smell.smell_type}

            Target Class:
            {target_class}

            Dependencies:

            {neighbors}

            Generate refactoring steps.

            Output JSON ONLY.

            [
            {{
                "operation":"SplitClass",
                "target_class":"...",
                "source_class":null,
                "affected_methods":["..."],
                "estimated_loc_impact":100,
                "rationale":"..."
            }}
            ]
        """

    def report(self) -> Dict[str, int]:
        return {
            "total_mutations": self.mutation_count,
            "accepted_mutations": self.accepted_mutations,
            "rejected_mutations": self.mutation_count - self.accepted_mutations
        }
