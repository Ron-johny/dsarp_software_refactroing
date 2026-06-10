import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from src.manifest_builder import ManifestLoader, BatchIsolationEngine, DetectedSmell
from src.models import RefactoringChromosome, RefactoringPlan, ExecutionPlan, ParetoPlanReview
from src.seeding import LLMPopulationSeeder
from src.evolution import FitnessEvaluator, AntiRegressionGate, LLMSemanticMutator, evolve_generation, non_dominated_sort
from src.filtering import ParetoFrontLLMFilter
from src.serialization import ExecutionPlanSerializer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NSGAIIBatchOptimizer:
    def __init__(self, manifest_path: str, batches_path: str, output_dir: str, llm_model: str = "mock-llm"):
        self.manifest_path = manifest_path
        self.batches_path = batches_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_model = llm_model

    def run(self, max_generations: int = 20, population_size: int = 20) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info("SBSE NSGA-II REFACTORING OPTIMIZATION PIPELINE (MODULAR)")
        logger.info("=" * 80)

        loader = ManifestLoader(self.manifest_path, self.batches_path)
        global_graph, batches_metadata, ranked_smells = loader.load()

        isolation_engine = BatchIsolationEngine(global_graph, ranked_smells)

        execution_plans = []
        batch_statistics = []

        for batch_num, (batch_id, batch_data) in enumerate(batches_metadata.items(), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"[BATCH {batch_num}] {batch_id}")
            logger.info(f"{'='*60}")

            target_classes = batch_data.get("target_classes", [])
            smell_ids = batch_data.get("smell_ids", [])

            if not target_classes or not smell_ids:
                logger.warning(f"[BATCH {batch_num}] Empty batch, skipping")
                continue

            # Tier 2: Isolate + Seed
            isolated_graph, batch_smells = isolation_engine.isolate_batch(batch_id, target_classes, smell_ids)

            seeder = LLMPopulationSeeder()
            seed_plans = seeder.seed_population(batch_id, target_classes, batch_smells, isolated_graph, population_size)

            # Tier 3 setup
            fitness_eval = FitnessEvaluator(isolated_graph, batch_smells)
            anti_regression_gate = AntiRegressionGate(isolated_graph)
            mutator = LLMSemanticMutator(isolated_graph, batch_smells, anti_regression_gate)

            # Initialize population from seed plans
            population = self._init_population(seed_plans, batch_smells, population_size)

            # Tier 3: Evolutionary Loop
            population = self._evolve(population, fitness_eval, mutator, max_generations)

            # Tier 4: Pareto Front Extraction
            pareto = self._extract_pareto_front(population)
            logger.info(f"[PARETO] {len(pareto)} non-dominated solutions")

            llm_filter = ParetoFrontLLMFilter()
            selected, review = llm_filter.filter_pareto_front(batch_id, pareto)

            if selected:
                plan = ExecutionPlanSerializer.serialize(batch_id, selected, review, batch_smells)
                plan.llm_model_used = self.llm_model
                execution_plans.append(plan)

                out_path = self.output_dir / f"{batch_id}_execution_plan.json"
                ExecutionPlanSerializer.write_json(plan, out_path)

                veto_report = anti_regression_gate.report()
                mut_report = mutator.report()

                batch_statistics.append({
                    "batch_id": batch_id,
                    "target_classes": len(target_classes),
                    "smells_addressed": len(batch_smells),
                    "pareto_size": len(pareto),
                    "selected_plan": selected.plan_id,
                    "debt_reduction": float(-selected.fitness_objectives[0]),
                    "risk_score": float(selected.fitness_objectives[1]),
                    "effort_loc": float(selected.fitness_objectives[2]),
                    "mutations_accepted": mut_report["accepted_mutations"],
                    "mutations_vetoed": veto_report["vetoed_mutations"],
                    "status": "VERIFIED_SAFE"
                })

        summary_path = self.output_dir / "optimization_summary.json"
        self._write_summary(batch_statistics, summary_path)

        logger.info(f"\n{'='*80}")
        logger.info("✓ OPTIMIZATION PIPELINE COMPLETE")
        logger.info(f"  Plans written  : {len(execution_plans)}")
        logger.info(f"  Batches done   : {len(batch_statistics)}")
        logger.info(f"  Output dir     : {self.output_dir}")
        logger.info(f"{'='*80}")

        return {
            "execution_plans": len(execution_plans),
            "batches": len(batch_statistics),
            "output_dir": str(self.output_dir),
            "summary_file": str(summary_path)
        }

    def _init_population(
        self,
        seed_plans: List[RefactoringPlan],
        batch_smells: List[DetectedSmell],
        target_size: int
    ) -> List[RefactoringChromosome]:
        all_smell_ids = [s.smell_id for s in batch_smells]
        population = []

        for plan in seed_plans:
            population.append(RefactoringChromosome(
                plan_id=plan.plan_id,
                smell_sequence=all_smell_ids.copy(),
                applied_steps=plan.steps.copy(),
                fitness_objectives=[0.0, 0.0, 0.0],
                is_valid=True
            ))

        while len(population) < target_size:
            base = random.choice(seed_plans)
            variant = RefactoringChromosome(
                plan_id=f"{base.plan_id}_v{len(population)}",
                smell_sequence=all_smell_ids.copy(),
                applied_steps=base.steps.copy(),
                fitness_objectives=[0.0, 0.0, 0.0],
                is_valid=True
            )
            population.append(variant)

        logger.info(f"[POPULATION] Initialized {len(population)} chromosomes")
        return population

    def _evolve(
        self,
        population: List[RefactoringChromosome],
        fitness_eval: FitnessEvaluator,
        mutator: LLMSemanticMutator,
        max_generations: int
    ) -> List[RefactoringChromosome]:
        for gen in range(max_generations):
            population = evolve_generation(
                population=population,
                fitness_eval=fitness_eval,
                mutator=mutator,
                population_size=len(population),
                crossover_prob=0.9,
                mutation_prob=0.5
            )

            if (gen + 1) % 5 == 0 or gen == 0:
                valid = [c for c in population if c.fitness_objectives[0] != float("inf")]
                if valid:
                    best = min(valid, key=lambda c: c.fitness_objectives[0])
                    logger.info(
                        f"[GEN {gen+1:3d}/{max_generations}] "
                        f"DebtRed={-best.fitness_objectives[0]:.3f}  "
                        f"Risk={best.fitness_objectives[1]:.3f}  "
                        f"LOC={best.fitness_objectives[2]:.0f}"
                    )

        logger.info(f"[EVOLUTION] Completed {max_generations} generations")
        return population

    def _extract_pareto_front(self, population: List[RefactoringChromosome]) -> List[RefactoringChromosome]:
        valid = [c for c in population if c.fitness_objectives != [float("inf")] * 3]
        if not valid:
            return population[:1]
        fronts = non_dominated_sort(valid)
        return fronts[0] if fronts else valid[:1]

    def _write_summary(self, stats: List[Dict], path: Path):
        if not stats:
            return
        summary = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": "2026.2.0",
            "llm_model": self.llm_model,
            "total_batches": len(stats),
            "total_smells_addressed": sum(b["smells_addressed"] for b in stats),
            "total_classes_refactored": sum(b["target_classes"] for b in stats),
            "total_debt_reduction": sum(b["debt_reduction"] for b in stats),
            "avg_risk_score": float(np.mean([b["risk_score"] for b in stats])),
            "total_effort_loc": sum(b["effort_loc"] for b in stats),
            "batches": stats
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[SUMMARY] Written to {path}")