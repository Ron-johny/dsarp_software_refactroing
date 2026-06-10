import json
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ClassMetrics:
    fqn: str
    loc: int
    wmc: int
    lcom: float
    cbo: int
    methods: list = field(default_factory=list)
    detected_smells: list = field(default_factory=list)
    file_path: str = ""


@dataclass
class DetectedSmell:
    smell_id: str
    smell_type: str
    level: str
    target_fqn: str
    description: str
    file_path: str
    impact_score: float
    batch_id: str


class ManifestLoader:

    def __init__(self, manifest_path, batches_path):
        self.manifest_path = Path(manifest_path)
        self.batches_path = Path(batches_path)

    def load(self):
        with open(self.manifest_path) as f:
            manifest = json.load(f)

        with open(self.batches_path) as f:
            batches = json.load(f)

        graph = nx.DiGraph()

        for node in manifest["system_graph"]["nodes"]:
            graph.add_node(
                node["fqn"],
                **node["metrics"],
                methods=node.get("methods", []),
                detected_smells=node.get("detected_smells", []),
                file_path=node.get("file_path", "")
            )

        for edge in manifest["system_graph"]["edges"]:
            graph.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1))

        return (
            graph,
            batches,
            manifest["prioritization_registry"]["ranked_smells"]
        )


class BatchIsolationEngine:

    def __init__(self, graph, ranked_smells):
        self.graph = graph
        self.smell_by_id = {s["smell_id"]: s for s in ranked_smells}

    def isolate_batch(self, batch_id, target_classes, smell_ids):
        nodes = set(target_classes)

        for cls in target_classes:
            if cls in self.graph:
                nodes.update(self.graph.successors(cls))
                nodes.update(self.graph.predecessors(cls))

        subgraph = self.graph.subgraph(nodes).copy()

        smells = []
        for sid in smell_ids:
            if sid in self.smell_by_id:
                s = self.smell_by_id[sid]
                smells.append(
                    DetectedSmell(
                        smell_id=s["smell_id"],
                        smell_type=s["type"],
                        level=s["level"],
                        target_fqn=s["target_class"],
                        description=s["description"],
                        file_path=s.get("file_path", ""),
                        impact_score=s["impact_score"],
                        batch_id=batch_id
                    )
                )

        return subgraph, smells