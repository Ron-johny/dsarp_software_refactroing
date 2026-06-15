# src/utils/graph_utils.py

import networkx as nx
from typing import Dict, Any, List


def get_structural_neighbors(graph: nx.DiGraph, target_class: str) -> Dict[str, Any]:
    if target_class not in graph:
        return {}
    return {
        "target_class": target_class,
        "outbound_dependencies": list(graph.successors(target_class)),
        "inbound_dependents": list(graph.predecessors(target_class)),
        "node_data": dict(graph.nodes[target_class])
    }


def simulate_refactoring(graph: nx.DiGraph, steps) -> nx.DiGraph:
    sim_graph = graph.copy()
    for step in steps:
        if step.operation == "ExtractInterface":
            if step.target_class in sim_graph.nodes:
                node = sim_graph.nodes[step.target_class]
                node["cbo"] = max(1, int(node.get("cbo", 1) * 0.8))
                node["wmc"] = max(1, int(node.get("wmc", 1) * 0.9))

        elif step.operation == "MoveMethod":
            if step.source_class and step.source_class in sim_graph.nodes and step.target_class in sim_graph.nodes:
                sim_graph.nodes[step.source_class]["cbo"] = max(1, sim_graph.nodes[step.source_class].get("cbo", 1) - 1)
                sim_graph.nodes[step.target_class]["cbo"] = sim_graph.nodes[step.target_class].get("cbo", 1) + 1
                sim_graph.nodes[step.source_class]["loc"] = max(
                    100,
                    sim_graph.nodes[step.source_class].get("loc", 100) - step.estimated_loc_impact
                )
                sim_graph.nodes[step.target_class]["loc"] = (
                    sim_graph.nodes[step.target_class].get("loc", 0) + step.estimated_loc_impact
                )

        elif step.operation == "SplitClass":
            if step.target_class in sim_graph.nodes:
                node = sim_graph.nodes[step.target_class]
                node["wmc"] = max(1, int(node.get("wmc", 1) * 0.6))
                node["loc"] = max(100, int(node.get("loc", 100) * 0.6))
                node["cbo"] = max(1, int(node.get("cbo", 1) * 0.7))

        elif step.operation == "IntroducePolymorphism":
            if step.target_class in sim_graph.nodes:
                node = sim_graph.nodes[step.target_class]
                node["cbo"] = max(1, int(node.get("cbo", 1) * 0.75))

    return sim_graph


def calculate_cbo(graph: nx.DiGraph) -> int:
    return sum(graph.nodes[node].get("cbo", 0) for node in graph.nodes())


def calculate_wmc(graph: nx.DiGraph) -> int:
    return sum(graph.nodes[node].get("wmc", 0) for node in graph.nodes())


def check_metric_violations(graph: nx.DiGraph, cbo_thresh=25, wmc_thresh=40, loc_thresh=1000) -> List[str]:
    violations = []
    for node_id, data in graph.nodes(data=True):
        if data.get("cbo", 0) > cbo_thresh:
            violations.append(f"{node_id}: CBO {data['cbo']} > {cbo_thresh}")
        if data.get("wmc", 0) > wmc_thresh:
            violations.append(f"{node_id}: WMC {data['wmc']} > {wmc_thresh}")
        if data.get("loc", 0) > loc_thresh:
            violations.append(f"{node_id}: LOC {data['loc']} > {loc_thresh}")
    return violations