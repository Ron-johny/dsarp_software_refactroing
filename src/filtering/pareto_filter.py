import json
import logging
from typing import List, Tuple, Optional
from pydantic import ValidationError

from src.models import RefactoringChromosome, ParetoPlanReview
from src.utils.mock_llm import review_pareto_front

logger = logging.getLogger(__name__)

class ParetoFrontLLMFilter:
    def filter_pareto_front(
        self,
        batch_id: str,
        pareto_solutions: List[RefactoringChromosome]
    ) -> Tuple[RefactoringChromosome, Optional[ParetoPlanReview]]:
        if not pareto_solutions:
            logger.error("[PARETO FILTER] No solutions to filter")
            return None, None

        plan_ids = [s.plan_id for s in pareto_solutions]
        review_json = review_pareto_front(batch_id, plan_ids)

        try:
            review = ParetoPlanReview(**json.loads(review_json))
            logger.info(f"[PARETO FILTER] Selected plan: {review.selected_plan_id}")

            selected = next(
                (s for s in pareto_solutions if s.plan_id == review.selected_plan_id),
                pareto_solutions[0]
            )
            return selected, review

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"[PARETO FILTER] Review parse error: {e}")
            return pareto_solutions[0], None
