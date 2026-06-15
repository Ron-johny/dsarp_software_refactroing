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
            node_id = node.get("fqn", node.get("id"))
            graph.add_node(
                node_id,
                **node["metrics"],
                methods=node.get("methods", []),
                detected_smells=node.get("detected_smells", []),
                file_path=node.get("file_path", "")
            )

        for edge in manifest["system_graph"]["edges"]:
            graph.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1))

        ranked_smells = manifest["prioritization_registry"]["ranked_smells"]

        # 🛠️ Intercept and rebuild the batch metadata with accurate smell IDs
        for batch_id, batch_data in batches.items():
            # Initialize a real list to hold the true smell IDs for this batch
            batch_data["smell_ids"] = []
            
        # Distribute smells to their respective batches based on execution_batch_id
        for smell in ranked_smells:
            bid = smell.get("execution_batch_id")
            if bid in batches:
                batches[bid]["smell_ids"].append(smell["smell_id"])

        # Clean up batches that didn't end up matching any smells in the registry
        processed_batches = {
            bid: bdata for bid, bdata in batches.items() 
            if bdata.get("smell_ids")
        }

        # Override total_smells with the true count
        for bid, bdata in processed_batches.items():
            bdata["total_smells"] = len(bdata["smell_ids"])

        return graph, processed_batches, ranked_smells


class BatchIsolationEngine:

    def __init__(self, graph, ranked_smells):
        self.graph = graph
        # Support fallback if the target property name varies across items
        self.smell_by_id = {}
        for s in ranked_smells:
            self.smell_by_id[s["smell_id"]] = s

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
                # Fallback handler for target class keys
                target_fqn = s.get("target", s.get("target_class", ""))
                
                smells.append(
                    DetectedSmell(
                        smell_id=s["smell_id"],
                        smell_type=s["type"],
                        level=s["level"],
                        target_fqn=target_fqn,
                        description=s["description"],
                        file_path=s.get("file_path", ""),
                        impact_score=s["impact_score"],
                        batch_id=batch_id
                    )
                )

        return subgraph, smells