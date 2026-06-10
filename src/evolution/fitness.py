import logging
import networkx as nx
from typing import List, Tuple

from src.models import RefactoringChromosome
from src.manifest_builder import DetectedSmell
from src.utils.graph_utils import calculate_cbo, calculate_wmc, simulate_refactoring

logger = logging.getLogger(__name__)

class FitnessEvaluator:
    SAFE_CBO = 25
    SAFE_WMC = 40
    SAFE_LOC = 1000

    def __init__(self, isolated_graph: nx.DiGraph, batch_smells: List[DetectedSmell]):
        self.isolated_graph = isolated_graph.copy()
        self.batch_smells = batch_smells
        self.smell_by_id = {s.smell_id: s for s in batch_smells}

        self.baseline_cbo = calculate_cbo(self.isolated_graph)
        self.baseline_wmc = calculate_wmc(self.isolated_graph)
        self.baseline_debt = sum(s.impact_score for s in batch_smells)

        logger.info(
            f"[FITNESS] Baseline: CBO={self.baseline_cbo}, WMC={self.baseline_wmc}, "
            f"Debt={self.baseline_debt:.3f}"
        )

    def evaluate(self, chromosome: RefactoringChromosome) -> Tuple[float, float, float]:
        """
        f1 = -debt_reduction  (minimize -> maximize debt reduction)
        f2 = structural risk delta (CBO+WMC change)
        f3 = LOC churn effort
        """
        sim_graph = simulate_refactoring(self.isolated_graph, chromosome.applied_steps)

        # f1: Technical Debt Reduction
        debt_resolved = sum(
            self.smell_by_id[sid].impact_score
            for sid in chromosome.smell_sequence
            if sid in self.smell_by_id
        )
        f1 = -debt_resolved

        # f2: Structural Risk
        sim_cbo = calculate_cbo(sim_graph)
        sim_wmc = calculate_wmc(sim_graph)
        cbo_delta = (sim_cbo - self.baseline_cbo) / max(self.baseline_cbo, 1)
        wmc_delta = (sim_wmc - self.baseline_wmc) / max(self.baseline_wmc, 1)
        f2 = max(0.0, (abs(cbo_delta) + abs(wmc_delta)) / 2.0)

        # f3: Effort (LOC)
        f3 = float(sum(max(0, s.estimated_loc_impact) for s in chromosome.applied_steps))

        chromosome.fitness_objectives = [f1, f2, f3]
        return f1, f2, f3
