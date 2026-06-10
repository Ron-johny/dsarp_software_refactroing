from src.optimizer import NSGAIIBatchOptimizer

"""
Just a mapping with previous work: Delete it later

| Tier / Phase                       | Step | Target File to Open                                                         |
| ---------------------------------- | ---- | --------------------------------------------------------------------------- |
| **Tier 1: Ingestion & Validation** | S1   | `src/models/refactoring.py`                                                 |
| **Tier 1: Ingestion & Validation** | S2   | `src/manifest_builder.py` and `src/utils/graph_utils.py`                    |
| **Tier 2: Optimization Setup**     | S3   | `src/optimizer.py` (or a dedicated class inside `src/utils/graph_utils.py`) |
| **Tier 2: Optimization Setup**     | S4   | `src/seeding/seeder.py`                                                     |
| **Tier 2: Optimization Setup**     | S5   | `src/models/chromosome.py` and `src/evolution/fitness.py`                   |
| **Tier 3: Evolutionary Loop**      | S8   | `src/evolution/nsga2.py`                                                    |
| **Tier 3: Evolutionary Loop**      | S7   | `src/evolution/mutator.py`                                                  |
| **Tier 3: Evolutionary Loop**      | S6   | `src/evolution/gate.py`                                                     |
| **Tier 4: Review & Hand-off**      | S9   | `src/filtering/pareto_filter.py`                                            |
| **Tier 4: Review & Hand-off**      | S10  | `src/filtering/pareto_filter.py`                                            |
| **Tier 4: Review & Hand-off**      | S11  | `src/serialization/serializer.py`                                           |s
"""


def main():
    optimizer = NSGAIIBatchOptimizer(
        manifest_path="data/nsga2_input_manifest.json",
        batches_path="data/nsga2_batches.json",
        output_dir="output"
    )

    optimizer.run(
        max_generations=20,
        population_size=20
    )


if __name__ == "__main__":
    main()