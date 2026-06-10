import json


def generate_seed_plans(batch_id: str, target_classes: list = None, smells: list = None) -> str:
    primary_class = (target_classes or ["org.apache.tika.parser.AutoDetectParser"])[0]
    smell_ids = [s.smell_id for s in (smells or [])][:3] or ["DESIGN_02443"]

    plans = [
        {
            "plan_id": f"PLAN_{batch_id}_001",
            "title": "Strategy Pattern Refactor",
            "target_classes": [primary_class],
            "smells_addressed": smell_ids,
            "design_pattern": "Strategy",
            "rationale": "Encapsulate behavior variations to reduce coupling and improve extensibility.",
            "steps": [
                {
                    "operation": "ExtractInterface",
                    "target_class": primary_class,
                    "source_class": None,
                    "affected_methods": ["detect", "process"],
                    "estimated_loc_impact": 120,
                    "rationale": "Expose abstraction to reduce coupling."
                },
                {
                    "operation": "MoveMethod",
                    "target_class": primary_class,
                    "source_class": primary_class,
                    "affected_methods": ["createStrategy"],
                    "estimated_loc_impact": 80,
                    "rationale": "Relocate factory logic to reduce WMC."
                }
            ],
            "estimated_effort": 10,
            "risk_level": "MEDIUM"
        },
        {
            "plan_id": f"PLAN_{batch_id}_002",
            "title": "Adapter Pattern for External Coupling",
            "target_classes": [primary_class],
            "smells_addressed": smell_ids,
            "design_pattern": "Adapter",
            "rationale": "Wrap external dependencies with adapters to isolate coupling.",
            "steps": [
                {
                    "operation": "IntroducePolymorphism",
                    "target_class": primary_class,
                    "source_class": None,
                    "affected_methods": ["parse"],
                    "estimated_loc_impact": 150,
                    "rationale": "Introduce abstract layer for polymorphic dispatch."
                }
            ],
            "estimated_effort": 8,
            "risk_level": "LOW"
        },
        {
            "plan_id": f"PLAN_{batch_id}_003",
            "title": "Decompose God Class via SplitClass",
            "target_classes": [primary_class],
            "smells_addressed": smell_ids,
            "design_pattern": "SRP",
            "rationale": "Split overloaded class into single-responsibility components.",
            "steps": [
                {
                    "operation": "SplitClass",
                    "target_class": primary_class,
                    "source_class": None,
                    "affected_methods": ["parse", "detect", "identify"],
                    "estimated_loc_impact": 300,
                    "rationale": "Reduce WMC and LOC by decomposing responsibilities."
                }
            ],
            "estimated_effort": 16,
            "risk_level": "HIGH"
        }
    ]
    return json.dumps(plans)


def generate_mutation_steps(smell_type: str, target_class: str = "TargetClass") -> str:
    steps_map = {
        "God Class": [
            {
                "operation": "SplitClass",
                "target_class": target_class,
                "source_class": None,
                "affected_methods": ["parse", "detect"],
                "estimated_loc_impact": 250,
                "rationale": "Reduce responsibility overload by splitting into SRP components."
            }
        ],
        "Unexploited Encapsulation": [
            {
                "operation": "ExtractInterface",
                "target_class": target_class,
                "source_class": None,
                "affected_methods": ["detect"],
                "estimated_loc_impact": 120,
                "rationale": "Expose public interface for better encapsulation."
            }
        ],
        "Missing Hierarchy": [
            {
                "operation": "IntroducePolymorphism",
                "target_class": target_class,
                "source_class": None,
                "affected_methods": ["parse"],
                "estimated_loc_impact": 200,
                "rationale": "Introduce abstract base class to enable polymorphism."
            }
        ],
        "Feature Envy": [
            {
                "operation": "MoveMethod",
                "target_class": target_class,
                "source_class": target_class,
                "affected_methods": ["compute"],
                "estimated_loc_impact": 90,
                "rationale": "Move method closer to the data it operates on."
            }
        ]
    }
    steps = steps_map.get(smell_type, [
        {
            "operation": "ExtractInterface",
            "target_class": target_class,
            "source_class": None,
            "affected_methods": ["process"],
            "estimated_loc_impact": 100,
            "rationale": "Generic structural improvement to reduce coupling."
        }
    ])
    return json.dumps(steps)


def review_pareto_front(batch_id: str = "BATCH_001", plan_ids: list = None) -> str:
    ids = plan_ids or ["PLAN_BATCH_001", "PLAN_BATCH_002"]
    review = {
        "reviewed_plan_ids": ids,
        "selected_plan_id": ids[0],
        "architectural_review_rationale": (
            "Selected plan provides the best tradeoff: highest debt reduction with acceptable "
            "structural risk and reasonable implementation effort. Staged rollout recommended."
        ),
        "maintainability_score": 0.87,
        "domain_safety_score": 0.91,
        "risk_mitigations": [
            "Stage rollout: ExtractInterface before MoveMethod",
            "Add integration tests for relocated methods",
            "Peer review before merging refactored classes"
        ],
        "verification_notes": (
            "All proposals pass CBO/WMC/LOC thresholds. No metric regressions detected in simulation."
        )
    }
    return json.dumps(review)