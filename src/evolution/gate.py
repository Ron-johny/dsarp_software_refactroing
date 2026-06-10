import logging
import networkx as nx
from typing import Dict

from src.models import RefactoringChromosome
from src.utils.graph_utils import simulate_refactoring, check_metric_violations

logger = logging.getLogger(__name__)

class AntiRegressionGate:
    SAFE_CBO_THRESHOLD = 25
    SAFE_WMC_THRESHOLD = 40
    SAFE_LOC_THRESHOLD = 1000

    def __init__(self, isolated_graph: nx.DiGraph):
        self.isolated_graph = isolated_graph
        self.veto_count = 0
        self.pass_count = 0

    def validate_mutation(self, chromosome: RefactoringChromosome) -> bool:
        try:
            sim_graph = simulate_refactoring(self.isolated_graph, chromosome.applied_steps)
            violations = check_metric_violations(
                sim_graph,
                self.SAFE_CBO_THRESHOLD,
                self.SAFE_WMC_THRESHOLD,
                self.SAFE_LOC_THRESHOLD
            )

            if violations:
                chromosome.is_valid = False
                chromosome.rejection_reason = f"Metric violations: {violations[:3]}"
                chromosome.fitness_objectives = [float("inf"), float("inf"), float("inf")]
                self.veto_count += 1
                logger.warning(f"[VETO GATE] Rejected: {chromosome.rejection_reason}")
                return False

            self.pass_count += 1
            return True

        except Exception as e:
            chromosome.is_valid = False
            chromosome.rejection_reason = f"Validation error: {e}"
            chromosome.fitness_objectives = [float("inf"), float("inf"), float("inf")]
            self.veto_count += 1
            return False

    def report(self) -> Dict[str, int]:
        return {
            "passed_validations": self.pass_count,
            "vetoed_mutations": self.veto_count,
            "total_evaluated": self.pass_count + self.veto_count
        }
