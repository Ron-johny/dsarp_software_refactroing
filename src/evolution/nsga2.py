# src/evolution/nsga2.py

import random
import logging
from typing import List, Tuple, Dict
import numpy as np

from src.models import RefactoringChromosome
from src.evolution.fitness import FitnessEvaluator
from src.evolution.mutator import LLMSemanticMutator

logger = logging.getLogger(__name__)

def dominates(a: RefactoringChromosome, b: RefactoringChromosome) -> bool:
    """Returns True if chromosome 'a' dominates chromosome 'b' (all objectives <= and at least one <)."""
    f_a = a.fitness_objectives
    f_b = b.fitness_objectives
    return (
        all(x <= y for x, y in zip(f_a, f_b)) and
        any(x < y for x, y in zip(f_a, f_b))
    )

def non_dominated_sort(population: List[RefactoringChromosome]) -> List[List[RefactoringChromosome]]:
    """Sorts population into non-dominated fronts. Assigns the 'rank' attribute to each chromosome (1-indexed)."""
    fronts: List[List[RefactoringChromosome]] = []
    dominated_by: Dict[int, List[int]] = {i: [] for i in range(len(population))}
    domination_count: Dict[int, int] = {i: 0 for i in range(len(population))}
    current_front: List[int] = []

    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if i == j:
                continue
            if dominates(p, q):
                dominated_by[i].append(j)
            elif dominates(q, p):
                domination_count[i] += 1
        if domination_count[i] == 0:
            p.rank = 1
            current_front.append(i)

    while current_front:
        front_chroms = [population[i] for i in current_front]
        fronts.append(front_chroms)
        next_front = []
        for i in current_front:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = len(fronts) + 1
                    next_front.append(j)
        current_front = next_front

    return fronts

def calculate_crowding_distance(front: List[RefactoringChromosome]) -> List[RefactoringChromosome]:
    """Calculates and assigns crowding distance to each chromosome in a front."""
    n = len(front)
    if n == 0:
        return front
    if n <= 2:
        for chrom in front:
            chrom.crowding_distance = float("inf")
        return front

    for chrom in front:
        chrom.crowding_distance = 0.0

    num_objectives = 3

    for obj_idx in range(num_objectives):
        # Sort front by current objective value
        front.sort(key=lambda c: c.fitness_objectives[obj_idx])
        
        # Boundary elements have infinite distance
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        f_min = front[0].fitness_objectives[obj_idx]
        f_max = front[-1].fitness_objectives[obj_idx]
        range_val = f_max - f_min if f_max != f_min else 1.0

        for k in range(1, n - 1):
            if np.isinf(front[k].crowding_distance):
                continue
            dist_diff = (
                front[k + 1].fitness_objectives[obj_idx] -
                front[k - 1].fitness_objectives[obj_idx]
            )
            front[k].crowding_distance += dist_diff / range_val

    return front

def binary_tournament_selection(population: List[RefactoringChromosome]) -> RefactoringChromosome:
    """Selects one chromosome using binary tournament selection based on rank and crowding distance."""
    c1, c2 = random.sample(population, 2)
    if c1.rank < c2.rank:
        return c1
    elif c2.rank < c1.rank:
        return c2
    else:
        return c1 if c1.crowding_distance >= c2.crowding_distance else c2

def crossover(p1: RefactoringChromosome, p2: RefactoringChromosome) -> RefactoringChromosome:
    """Performs single-point crossover on the smell sequences of two chromosomes."""
    if len(p1.smell_sequence) < 2:
        return p1.copy()
    point = random.randint(1, len(p1.smell_sequence) - 1)
    child_smells = p1.smell_sequence[:point] + p2.smell_sequence[point:]
    
    # New child inherits steps from p1 (up to the point) and p2 (after the point)
    # matching the crossover logic of the smell sequence.
    step_point = int(point / len(p1.smell_sequence) * len(p1.applied_steps)) if p1.applied_steps else 0
    p2_step_point = int(point / len(p2.smell_sequence) * len(p2.applied_steps)) if p2.applied_steps else 0
    child_steps = p1.applied_steps[:step_point] + p2.applied_steps[p2_step_point:]
    
    return RefactoringChromosome(
        plan_id=f"{p1.plan_id}_x_{p2.plan_id[:8]}",
        smell_sequence=child_smells,
        applied_steps=child_steps,
        fitness_objectives=[0.0, 0.0, 0.0],
        is_valid=True
    )

def evolve_generation(
    population: List[RefactoringChromosome],
    fitness_eval: FitnessEvaluator,
    mutator: LLMSemanticMutator,
    population_size: int,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.5
) -> List[RefactoringChromosome]:
    """Runs one complete NSGA-II generation (selection, crossover, mutation, survival)."""
    # 1. Ensure fitness of all individuals is evaluated
    for chrom in population:
        if chrom.fitness_objectives == [0.0, 0.0, 0.0]:
            fitness_eval.evaluate(chrom)

    # 2. Rank and calculate crowding distance for the current parent population
    fronts = non_dominated_sort(population)
    for front in fronts:
        calculate_crowding_distance(front)

    # 3. Create Offspring Population
    offspring: List[RefactoringChromosome] = []
    while len(offspring) < population_size:
        p1 = binary_tournament_selection(population)
        p2 = binary_tournament_selection(population)

        if random.random() < crossover_prob:
            child = crossover(p1, p2)
        else:
            child = p1.copy()

        if random.random() < mutation_prob:
            child = mutator.mutate(child)

        fitness_eval.evaluate(child)
        offspring.append(child)

    # 4. Combine parents and offspring
    combined = population + offspring

    # 5. Non-dominated sort of combined population
    combined_fronts = non_dominated_sort(combined)

    # 6. Environmental Selection (Survival)
    next_population: List[RefactoringChromosome] = []
    for front in combined_fronts:
        calculate_crowding_distance(front)
        if len(next_population) + len(front) <= population_size:
            next_population.extend(front)
        else:
            # Sort by crowding distance in descending order
            front.sort(key=lambda c: c.crowding_distance, reverse=True)
            remaining_slots = population_size - len(next_population)
            next_population.extend(front[:remaining_slots])
            break

    return next_population
