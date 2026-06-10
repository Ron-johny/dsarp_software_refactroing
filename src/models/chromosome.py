from dataclasses import dataclass, field
from typing import List
from src.models.refactoring import RefactoringStep

@dataclass
class RefactoringChromosome:
    plan_id: str
    smell_sequence: List[str]
    applied_steps: List[RefactoringStep] = field(default_factory=list)
    fitness_objectives: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    is_valid: bool = True
    rejection_reason: str = ""
    rank: int = 0
    crowding_distance: float = 0.0

    def copy(self) -> "RefactoringChromosome":
        return RefactoringChromosome(
            plan_id=self.plan_id,
            smell_sequence=self.smell_sequence.copy(),
            applied_steps=[RefactoringStep(**s.dict()) for s in self.applied_steps],
            fitness_objectives=self.fitness_objectives.copy(),
            is_valid=self.is_valid,
            rejection_reason=self.rejection_reason,
            rank=self.rank,
            crowding_distance=self.crowding_distance
        )
