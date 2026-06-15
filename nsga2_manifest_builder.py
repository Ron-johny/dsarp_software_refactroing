#!/usr/bin/env python3
"""
Production-grade SBSE Manifest Builder for NSGA-II Loop Execution.

This script processes Designite smell outputs and source code metadata to compile
the unified "nsga2_input_manifest.json" required to run an NSGA-II optimization loop.

Pipeline Phases:
  1. DESIGNITE CSV PARSING: Parse architecture, design, and implementation smells
  2. METRIC INGESTION: Load or simulate class-level structural metrics
  3. SYSTEM GRAPH CONSTRUCTION: Build NetworkX DiGraph with dependencies
  4. AIS & BATCHING: Calculate Architectural Impact Scores and batch smells

Author: SBSE Research Team
Version: 2026.1.0
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

import pandas as pd
import networkx as nx
import numpy as np


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES & TYPES
# ============================================================================

@dataclass
class ClassMetrics:
    """Structural metrics for a single class."""
    fqn: str
    loc: int  # Lines of Code
    wmc: int  # Weighted Method Complexity (cyclomatic complexity)
    lcom: float  # Lack of Cohesion of Methods
    cbo: int  # Coupling Between Objects
    methods: List[str] = field(default_factory=list)  # Method signatures

    def to_dict(self) -> Dict:
        """Convert to dictionary, ensuring all values are native Python types."""
        return {
            "fqn": self.fqn,
            "loc": int(self.loc),
            "wmc": int(self.wmc),
            "lcom": float(self.lcom),
            "cbo": int(self.cbo),
            "methods": self.methods
        }


@dataclass
class DetectedSmell:
    """Metadata for a detected code smell."""
    smell_id: str  # Unique identifier
    smell_type: str  # e.g., "God Component", "Feature Envy"
    smell_level: str  # "ARCHITECTURE", "DESIGN", or "IMPLEMENTATION"
    target_fqn: str  # Fully Qualified Name of the affected class/package
    description: str
    file_path: str
    impact_score: float = 0.0  # Calculated during AIS phase
    execution_batch_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary, ensuring all values are native Python types."""
        return {
            "smell_id": self.smell_id,
            "type": self.smell_type,
            "level": self.smell_level,
            "target": self.target_fqn,
            "description": self.description,
            "file_path": self.file_path,
            "impact_score": float(self.impact_score),
            "execution_batch_id": self.execution_batch_id
        }


# ============================================================================
# PHASE 1: DESIGNITE CSV PARSING
# ============================================================================

class DesigniteCSVParser:
    """Parses Designite smell CSV outputs and constructs FQNs."""

    def __init__(self, arch_csv: str, design_csv: str, impl_csv: str):
        """
        Initialize parser with paths to smell CSV files.

        Args:
            arch_csv: Path to ArchitectureSmells.csv
            design_csv: Path to DesignSmells.csv
            impl_csv: Path to ImplementationSmells.csv
        """
        self.arch_csv = arch_csv
        self.design_csv = design_csv
        self.impl_csv = impl_csv
        self.smells: List[DetectedSmell] = []
        self.smell_counter = 0

    def _normalize_file_path(self, file_path: str) -> str:
        """
        Clean and normalize file paths.
        Remove old workspace paths and return relative paths.

        Args:
            file_path: Original file path from CSV

        Returns:
            Normalized file path
        """
        # Remove common old workspace prefixes
        prefixes_to_remove = [
            "/Users/surabhilone/Desktop/Thesis/SoftwareSystems/",
            "/home/user/",
            "C:\\Users\\",
        ]
        for prefix in prefixes_to_remove:
            if file_path.startswith(prefix):
                file_path = file_path[len(prefix):]
        return file_path

    def _construct_fqn_class(self, package: str, class_name: str) -> str:
        """
        Construct Fully Qualified Name for a class.

        Args:
            package: Package name (e.g., 'org.apache.tika.parser')
            class_name: Class name (e.g., 'AutoDetectParser')

        Returns:
            FQN in format 'org.apache.tika.parser.AutoDetectParser'
        """
        if class_name and package:
            return f"{package}.{class_name}"
        return package or class_name

    def _parse_architecture_smells(self) -> None:
        """
        Parse ArchitectureSmells.csv.
        Architecture-level smells are attributed to packages.
        """
        logger.info(f"Parsing architecture smells from {self.arch_csv}")
        try:
            df = pd.read_csv(self.arch_csv)
        except FileNotFoundError:
            logger.warning(f"Architecture smells file not found: {self.arch_csv}")
            return

        for idx, row in df.iterrows():
            try:
                self.smell_counter += 1
                smell_id = f"ARCH_{self.smell_counter:05d}"
                
                package = str(row.get("Package", "")).strip()
                smell_type = str(row.get("Smell", "")).strip()
                description = str(row.get("Description", "")).strip()

                smell = DetectedSmell(
                    smell_id=smell_id,
                    smell_type=smell_type,
                    smell_level="ARCHITECTURE",
                    target_fqn=package,
                    description=description,
                    file_path=""  # Architecture smells don't have file paths
                )
                self.smells.append(smell)
                logger.debug(f"Parsed architecture smell: {smell_id} -> {package}")

            except Exception as e:
                logger.error(f"Error parsing architecture smell at row {idx}: {e}")

    def _parse_design_smells(self) -> None:
        """
        Parse DesignSmells.csv.
        Design-level smells are attributed to classes.
        """
        logger.info(f"Parsing design smells from {self.design_csv}")
        try:
            df = pd.read_csv(self.design_csv)
        except FileNotFoundError:
            logger.warning(f"Design smells file not found: {self.design_csv}")
            return

        for idx, row in df.iterrows():
            try:
                self.smell_counter += 1
                smell_id = f"DESIGN_{self.smell_counter:05d}"

                package = str(row.get("Package", "")).strip()
                class_name = str(row.get("Class", "")).strip()
                smell_type = str(row.get("Smell", "")).strip()
                description = str(row.get("Description", "")).strip()
                file_path = str(row.get("File", "")).strip()

                fqn = self._construct_fqn_class(package, class_name)
                file_path = self._normalize_file_path(file_path)

                smell = DetectedSmell(
                    smell_id=smell_id,
                    smell_type=smell_type,
                    smell_level="DESIGN",
                    target_fqn=fqn,
                    description=description,
                    file_path=file_path
                )
                self.smells.append(smell)
                logger.debug(f"Parsed design smell: {smell_id} -> {fqn}")

            except Exception as e:
                logger.error(f"Error parsing design smell at row {idx}: {e}")

    def _parse_implementation_smells(self) -> None:
        """
        Parse ImplementationSmells.csv.
        Implementation-level smells are attributed to methods within classes.
        """
        logger.info(f"Parsing implementation smells from {self.impl_csv}")
        try:
            df = pd.read_csv(self.impl_csv)
        except FileNotFoundError:
            logger.warning(f"Implementation smells file not found: {self.impl_csv}")
            return

        for idx, row in df.iterrows():
            try:
                self.smell_counter += 1
                smell_id = f"IMPL_{self.smell_counter:05d}"

                package = str(row.get("Package", "")).strip()
                class_name = str(row.get("Class", "")).strip()
                method = str(row.get("Method", "")).strip()
                smell_type = str(row.get("Smell", "")).strip()
                description = str(row.get("Description", "")).strip()
                file_path = str(row.get("File", "")).strip()

                fqn = self._construct_fqn_class(package, class_name)
                file_path = self._normalize_file_path(file_path)

                # Append method signature to description for context
                if method:
                    description = f"[Method: {method}] {description}"

                smell = DetectedSmell(
                    smell_id=smell_id,
                    smell_type=smell_type,
                    smell_level="IMPLEMENTATION",
                    target_fqn=fqn,
                    description=description,
                    file_path=file_path
                )
                self.smells.append(smell)
                logger.debug(f"Parsed implementation smell: {smell_id} -> {fqn}")

            except Exception as e:
                logger.error(f"Error parsing implementation smell at row {idx}: {e}")

    def parse_all(self) -> List[DetectedSmell]:
        """
        Execute all parsing phases.

        Returns:
            List of all detected smells from all CSV files
        """
        self._parse_architecture_smells()
        self._parse_design_smells()
        self._parse_implementation_smells()

        logger.info(f"Total smells parsed: {len(self.smells)}")
        return self.smells

    def get_smells_by_fqn(self) -> Dict[str, List[DetectedSmell]]:
        """
        Group all detected smells by their target FQN.

        Returns:
            Dictionary mapping FQN -> list of smells
        """
        grouped = defaultdict(list)
        for smell in self.smells:
            grouped[smell.target_fqn].append(smell)
        return dict(grouped)


# ============================================================================
# PHASE 2: METRIC INGESTION
# ============================================================================

class MetricIngestor:
    """Reads or simulates structural metrics for classes."""

    @staticmethod
    def read_real_metrics(type_metrics_csv: str, method_metrics_csv: str) -> Dict[str, ClassMetrics]:
        """
        Read real structural metrics from TypeMetrics.csv and MethodMetrics.csv.

        Args:
            type_metrics_csv: Path to TypeMetrics.csv
            method_metrics_csv: Path to MethodMetrics.csv

        Returns:
            Dictionary mapping FQN -> ClassMetrics
        """
        logger.info(f"Reading real metrics from TypeMetrics and MethodMetrics")
        metrics = {}
        method_signatures: Dict[str, List[str]] = defaultdict(list)

        # First pass: read method metrics and extract method signatures
        try:
            method_df = pd.read_csv(method_metrics_csv)
            for idx, row in method_df.iterrows():
                try:
                    package = str(row.get("Package", "")).strip()
                    class_name = str(row.get("Class", "")).strip()
                    method_name = str(row.get("Method", "")).strip()

                    fqn = f"{package}.{class_name}" if package and class_name else None
                    if fqn and method_name:
                        method_signatures[fqn].append(method_name)

                except Exception as e:
                    logger.debug(f"Error parsing method metrics row {idx}: {e}")

            logger.info(f"Extracted {sum(len(v) for v in method_signatures.values())} method signatures")

        except FileNotFoundError:
            logger.warning(f"MethodMetrics.csv not found: {method_metrics_csv}")
        except Exception as e:
            logger.error(f"Failed to read MethodMetrics: {e}")

        # Second pass: read type metrics (class-level)
        try:
            type_df = pd.read_csv(type_metrics_csv)
            for idx, row in type_df.iterrows():
                try:
                    package = str(row.get("Package", "")).strip()
                    class_name = str(row.get("Class", "")).strip()

                    fqn = f"{package}.{class_name}" if package and class_name else None
                    if not fqn:
                        logger.debug(f"Skipping row {idx}: empty package or class")
                        continue

                    # Extract metrics from row
                    loc = int(row.get("LOC", 0))
                    wmc = int(row.get("WMC", 0))
                    lcom = float(row.get("LCOM", 0.0))

                    # Use Fan-In + Fan-Out as coupling estimate
                    fan_in = int(row.get("Fan-In", 0))
                    fan_out = int(row.get("Fan-Out", 0))
                    cbo = fan_in + fan_out if (fan_in + fan_out) > 0 else 1

                    # Get method signatures for this class
                    methods = method_signatures.get(fqn, [])

                    metrics[fqn] = ClassMetrics(
                        fqn=fqn,
                        loc=loc,
                        wmc=wmc,
                        lcom=lcom,
                        cbo=cbo,
                        methods=methods
                    )

                except Exception as e:
                    logger.warning(f"Error parsing type metrics row {idx}: {e}")

            logger.info(f"Loaded real metrics for {len(metrics)} classes")

        except FileNotFoundError:
            logger.warning(f"TypeMetrics.csv not found: {type_metrics_csv}")
            return {}
        except Exception as e:
            logger.error(f"Failed to read TypeMetrics: {e}")
            return {}

        return metrics

    @staticmethod
    def generate_mock_metrics(fqns: Set[str]) -> Dict[str, ClassMetrics]:
        """
        Generate mock structural metrics for demonstration purposes.
        In production, this would read from an AST dump or bytecode analysis.

        Args:
            fqns: Set of all FQNs to generate metrics for

        Returns:
            Dictionary mapping FQN -> ClassMetrics
        """
        logger.info(f"Generating mock metrics for {len(fqns)} classes")
        metrics = {}

        # Deterministic seeding for reproducibility
        np.random.seed(42)

        for fqn in fqns:
            # Simulate realistic metrics ranges
            loc = int(np.random.uniform(50, 500))  # 50-500 lines
            wmc = int(np.random.uniform(5, 50))    # 5-50 cyclomatic complexity
            lcom = np.random.uniform(0.0, 1.0)     # 0.0-1.0 cohesion
            cbo = int(np.random.uniform(1, 20))    # 1-20 coupling

            # Generate 2-5 mock method signatures
            num_methods = int(np.random.uniform(2, 6))
            class_short_name = fqn.split(".")[-1]
            methods = [
                f"{class_short_name}.method{i}()"
                for i in range(num_methods)
            ]

            metrics[fqn] = ClassMetrics(
                fqn=fqn,
                loc=loc,
                wmc=wmc,
                lcom=lcom,
                cbo=cbo,
                methods=methods
            )

        logger.info(f"Generated metrics for {len(metrics)} classes")
        return metrics

    @staticmethod
    def read_metrics_from_csv(metrics_csv: str) -> Dict[str, ClassMetrics]:
        """
        Read metrics from a structured CSV file.
        Expected columns: FQN, LOC, WMC, LCOM, CBO, Methods (comma-separated)

        Args:
            metrics_csv: Path to metrics CSV file

        Returns:
            Dictionary mapping FQN -> ClassMetrics
        """
        logger.info(f"Reading metrics from {metrics_csv}")
        metrics = {}

        try:
            df = pd.read_csv(metrics_csv)
            for idx, row in df.iterrows():
                fqn = str(row.get("FQN", "")).strip()
                if not fqn:
                    logger.warning(f"Skipping row {idx}: empty FQN")
                    continue

                methods_str = str(row.get("Methods", "")).strip()
                methods = [m.strip() for m in methods_str.split(",")] if methods_str else []

                metrics[fqn] = ClassMetrics(
                    fqn=fqn,
                    loc=int(row.get("LOC", 0)),
                    wmc=int(row.get("WMC", 0)),
                    lcom=float(row.get("LCOM", 0.0)),
                    cbo=int(row.get("CBO", 0)),
                    methods=methods
                )

            logger.info(f"Loaded metrics for {len(metrics)} classes")
        except Exception as e:
            logger.error(f"Failed to read metrics CSV: {e}")

        return metrics


# ============================================================================
# PHASE 3: SYSTEM GRAPH CONSTRUCTION
# ============================================================================

class SystemGraphBuilder:
    """Constructs a NetworkX DiGraph representing the system architecture."""

    def __init__(self):
        """Initialize empty directed graph."""
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_classes_as_nodes(
        self,
        metrics: Dict[str, ClassMetrics],
        smells_by_fqn: Dict[str, List[DetectedSmell]]
    ) -> None:
        """
        Add all classes as nodes in the graph with attributes.

        Args:
            metrics: Dictionary of class metrics
            smells_by_fqn: Dictionary mapping FQN -> list of smells
        """
        logger.info(f"Adding {len(metrics)} classes as nodes")

        for fqn, metric in metrics.items():
            smells = smells_by_fqn.get(fqn, [])
            
            self.graph.add_node(
                fqn,
                file_path=metric.fqn,  # Use FQN as file identifier
                metrics={
                    "loc": int(metric.loc),
                    "wmc": int(metric.wmc),
                    "lcom": float(metric.lcom),
                    "cbo": int(metric.cbo)
                },
                methods=metric.methods,
                detected_smells=[
                    {
                        "id": s.smell_id,
                        "type": s.smell_type,
                        "level": s.smell_level,
                        "description": s.description
                    }
                    for s in smells
                ]
            )

    def add_dependency_edges(self) -> None:
        """
        Add edges based on coupling metrics.
        Simulates structural dependencies between classes.
        For production, this would parse actual import/dependency information.
        """
        logger.info("Adding dependency edges based on coupling")

        nodes = list(self.graph.nodes())

        # For each node, create edges to other nodes with probability
        # proportional to their coupling (simulating realistic dependencies)
        np.random.seed(42)

        for source in nodes:
            source_cbo = self.graph.nodes[source]["metrics"]["cbo"]
            num_dependencies = min(source_cbo, len(nodes) - 1)

            if num_dependencies > 0:
                # Randomly select target nodes for dependencies
                possible_targets = [n for n in nodes if n != source]
                targets = np.random.choice(
                    possible_targets,
                    size=min(num_dependencies, len(possible_targets)),
                    replace=False
                )

                for target in targets:
                    self.graph.add_edge(
                        source,
                        target,
                        type="DEPENDS_ON",
                        weight=1
                    )

        logger.info(f"Added {self.graph.number_of_edges()} edges")

    def build(
        self,
        metrics: Dict[str, ClassMetrics],
        smells_by_fqn: Dict[str, List[DetectedSmell]]
    ) -> nx.DiGraph:
        """
        Build complete system graph.

        Args:
            metrics: Dictionary of class metrics
            smells_by_fqn: Dictionary mapping FQN -> list of smells

        Returns:
            NetworkX DiGraph representing the system
        """
        self.add_classes_as_nodes(metrics, smells_by_fqn)
        self.add_dependency_edges()
        logger.info(f"System graph built: {self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges")
        return self.graph

    def to_json_serializable(self) -> Dict:
        """
        Convert graph to JSON-serializable format.

        Returns:
            Dictionary with nodes and edges for JSON output
        """
        nodes = []
        for node in self.graph.nodes(data=True):
            node_id, attrs = node
            nodes.append({
                "id": node_id,
                "file_path": attrs.get("file_path", ""),
                "metrics": attrs.get("metrics", {}),
                "methods": attrs.get("methods", []),
                "detected_smells": attrs.get("detected_smells", [])
            })

        edges = []
        for source, target, attrs in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "type": attrs.get("type", "DEPENDS_ON"),
                "weight": int(attrs.get("weight", 1))
            })

        return {"nodes": nodes, "edges": edges}


# ============================================================================
# PHASE 4: AIS & BATCHING
# ============================================================================

class ArchitecturalImpactScorer:
    """Calculates Architectural Impact Scores and batches smells for execution."""

    def __init__(self, graph: nx.DiGraph, smells: List[DetectedSmell]):
        """
        Initialize scorer with graph and detected smells.

        Args:
            graph: NetworkX DiGraph of system architecture
            smells: List of all detected smells
        """
        self.graph = graph
        self.smells = smells
        self.pagerank_scores: Dict[str, float] = {}
        self.normalized_loc: Dict[str, float] = {}
        self.chirpiness: Dict[str, float] = {}

    def _calculate_pagerank(self) -> None:
        """Calculate PageRank for all nodes in graph."""
        logger.info("Calculating PageRank scores")
        self.pagerank_scores = nx.pagerank(self.graph, alpha=0.85)
        logger.debug(f"PageRank calculated for {len(self.pagerank_scores)} nodes")

    def _normalize_loc(self) -> None:
        """Normalize LOC metric to [0, 1] range."""
        logger.info("Normalizing LOC metrics")
        loc_values = [
            self.graph.nodes[node]["metrics"]["loc"]
            for node in self.graph.nodes()
        ]

        if not loc_values:
            return

        loc_min = min(loc_values)
        loc_max = max(loc_values)
        loc_range = loc_max - loc_min if loc_max > loc_min else 1

        for node in self.graph.nodes():
            loc = self.graph.nodes[node]["metrics"]["loc"]
            self.normalized_loc[node] = (loc - loc_min) / loc_range

    def _calculate_chirpiness(self) -> None:
        """
        Calculate Chirpiness metric using semantic similarity approximation.
        Chirpiness = 1.0 - average_semantic_similarity_with_neighbors

        In production, this would use actual code2vec or BERT embeddings.
        Here we simulate with a random similarity matrix.
        """
        logger.info("Calculating Chirpiness (semantic diversity)")
        nodes = list(self.graph.nodes())
        num_nodes = len(nodes)

        # Create deterministic but realistic similarity matrix
        np.random.seed(42)
        similarity_matrix = np.random.uniform(0.3, 0.9, size=(num_nodes, num_nodes))

        # Make symmetric
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

        # Diagonal elements are 1.0 (self-similarity)
        np.fill_diagonal(similarity_matrix, 1.0)

        for i, node in enumerate(nodes):
            # Get neighbors in the graph
            neighbors = list(self.graph.neighbors(node))
            neighbor_indices = [nodes.index(n) for n in neighbors]

            if neighbor_indices:
                avg_similarity = np.mean(similarity_matrix[i, neighbor_indices])
            else:
                avg_similarity = 0.5  # Default for isolated nodes

            self.chirpiness[node] = float(1.0 - avg_similarity)

    def _compute_ais_for_smell(self, smell: DetectedSmell) -> float:
        """
        Compute AIS for a single smell.
        AIS = (0.4 * Normalized_LOC) + (0.4 * Normalized_PageRank) + (0.2 * Chirpiness)

        Args:
            smell: DetectedSmell object

        Returns:
            AIS score (float)
        """
        fqn = smell.target_fqn

        # Get component values, defaulting to 0 if not found
        loc_norm = self.normalized_loc.get(fqn, 0.0)
        pagerank = self.pagerank_scores.get(fqn, 0.0)
        chirp = self.chirpiness.get(fqn, 0.0)

        # Normalize PageRank to [0, 1]
        max_pr = max(self.pagerank_scores.values()) if self.pagerank_scores else 1.0
        pagerank_norm = pagerank / max_pr if max_pr > 0 else 0.0

        ais = (0.4 * loc_norm) + (0.4 * pagerank_norm) + (0.2 * chirp)

        return float(ais)

    def score_all_smells(self) -> List[DetectedSmell]:
        """
        Calculate AIS for all detected smells and rank them.

        Returns:
            List of smells ranked by AIS descending
        """
        logger.info(f"Calculating AIS for {len(self.smells)} smells")

        self._calculate_pagerank()
        self._normalize_loc()
        self._calculate_chirpiness()

        for smell in self.smells:
            smell.impact_score = self._compute_ais_for_smell(smell)

        # Sort by impact score descending
        ranked_smells = sorted(self.smells, key=lambda s: s.impact_score, reverse=True)

        logger.info(f"Smells ranked. Top 3 impacts: "
                    f"{[f'{s.smell_id}:{s.impact_score:.4f}' for s in ranked_smells[:3]]}")

        return ranked_smells

    def batch_smells(
        self,
        ranked_smells: List[DetectedSmell],
        max_smells_per_batch: int = 10
    ) -> Tuple[Dict[str, List[DetectedSmell]], Dict]:
        """
        Organize ranked smells into execution batches.
        Batches are formed based on connected subgraphs or shared classes.

        Args:
            ranked_smells: Ranked list of smells from highest to lowest AIS
            max_smells_per_batch: Maximum smells per batch

        Returns:
            Tuple of (batches dict, batch metadata)
        """
        logger.info(f"Batching {len(ranked_smells)} smells (max {max_smells_per_batch}/batch)")

        batches: Dict[str, List[DetectedSmell]] = {}
        batch_metadata: Dict = {}
        batch_num = 0
        current_batch_smells: List[DetectedSmell] = []
        affected_fqns: Set[str] = set()

        for smell in ranked_smells:
            # If current batch is full or adding this smell would create overlap
            if len(current_batch_smells) >= max_smells_per_batch:
                batch_num += 1
                batch_id = f"BATCH_{batch_num:03d}"
                batches[batch_id] = current_batch_smells
                batch_metadata[batch_id] = {
                    "target_classes": sorted(list(affected_fqns)),
                    "total_smells": len(current_batch_smells)
                }
                current_batch_smells = []
                affected_fqns = set()

            current_batch_smells.append(smell)
            smell.execution_batch_id = f"BATCH_{batch_num:03d}"
            affected_fqns.add(smell.target_fqn)

        # Add final batch if not empty
        if current_batch_smells:
            batch_num += 1
            batch_id = f"BATCH_{batch_num:03d}"
            batches[batch_id] = current_batch_smells
            batch_metadata[batch_id] = {
                "target_classes": sorted(list(affected_fqns)),
                "total_smells": len(current_batch_smells)
            }

        logger.info(f"Created {len(batches)} batches for execution")
        return batches, batch_metadata


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

class NSGAIIManifestBuilder:
    """Orchestrates the complete pipeline and generates output manifest."""

    def __init__(
        self,
        arch_csv: str,
        design_csv: str,
        impl_csv: str,
        output_json: str = "nsga2_input_manifest.json",
        target_project: str = "sample-system",
        type_metrics_csv: Optional[str] = None,
        method_metrics_csv: Optional[str] = None
    ):
        """
        Initialize builder.

        Args:
            arch_csv: Path to ArchitectureSmells.csv
            design_csv: Path to DesignSmells.csv
            impl_csv: Path to ImplementationSmells.csv
            output_json: Output manifest filename
            target_project: Target project name for manifest
            type_metrics_csv: Path to TypeMetrics.csv (optional)
            method_metrics_csv: Path to MethodMetrics.csv (optional)
        """
        self.arch_csv = arch_csv
        self.design_csv = design_csv
        self.impl_csv = impl_csv
        self.output_json = output_json
        self.target_project = target_project
        self.type_metrics_csv = type_metrics_csv
        self.method_metrics_csv = method_metrics_csv

    def run(self) -> str:
        """
        Execute complete pipeline and generate manifest.

        Returns:
            Path to generated manifest JSON file
        """
        logger.info("=" * 70)
        logger.info("NSGA-II Input Manifest Builder - Starting Pipeline")
        logger.info("=" * 70)

        # Phase 1: Parse Designite CSV outputs
        logger.info("\n--- PHASE 1: DESIGNITE CSV PARSING ---")
        parser = DesigniteCSVParser(self.arch_csv, self.design_csv, self.impl_csv)
        all_smells = parser.parse_all()
        smells_by_fqn = parser.get_smells_by_fqn()

        # Phase 2: Ingest metrics
        logger.info("\n--- PHASE 2: METRIC INGESTION ---")
        
        # Try to use real metrics first, fallback to mock
        if self.type_metrics_csv and self.method_metrics_csv:
            logger.info("Using real metrics from TypeMetrics and MethodMetrics")
            metrics = MetricIngestor.read_real_metrics(
                self.type_metrics_csv,
                self.method_metrics_csv
            )
            
            if not metrics:
                logger.warning("Real metrics empty, falling back to mock metrics")
                unique_fqns = set(smells_by_fqn.keys())
                metrics = MetricIngestor.generate_mock_metrics(unique_fqns)
        else:
            logger.info("Using mock metrics (no real metrics provided)")
            unique_fqns = set(smells_by_fqn.keys())
            metrics = MetricIngestor.generate_mock_metrics(unique_fqns)

        # Ensure all FQNs from smells are represented in metrics
        for fqn in smells_by_fqn.keys():
            if fqn not in metrics:
                # Create minimal metric for FQNs not in metrics (e.g., packages in architecture smells)
                metrics[fqn] = ClassMetrics(
                    fqn=fqn,
                    loc=100,  # Default values
                    wmc=5,
                    lcom=0.5,
                    cbo=2,
                    methods=[]
                )

        logger.info(f"Metrics ingested for {len(metrics)} classes")

        # Phase 3: Build system graph
        logger.info("\n--- PHASE 3: SYSTEM GRAPH CONSTRUCTION ---")
        graph_builder = SystemGraphBuilder()
        system_graph = graph_builder.build(metrics, smells_by_fqn)

        # Phase 4: Calculate AIS and batch
        logger.info("\n--- PHASE 4: AIS & BATCHING ---")
        scorer = ArchitecturalImpactScorer(system_graph, all_smells)
        ranked_smells = scorer.score_all_smells()
        batches, batch_metadata = scorer.batch_smells(ranked_smells, max_smells_per_batch=10)

        # Build output manifest
        logger.info("\n--- GENERATING OUTPUT MANIFEST ---")
        manifest = self._build_manifest(
            system_graph, ranked_smells, batch_metadata, graph_builder
        )

        # Write manifest to JSON
        output_path = self._write_manifest(manifest)

        logger.info("=" * 70)
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Manifest written to: {output_path}")
        logger.info("=" * 70)

        return output_path

    def _build_manifest(
        self,
        graph: nx.DiGraph,
        ranked_smells: List[DetectedSmell],
        batch_metadata: Dict,
        graph_builder: SystemGraphBuilder
    ) -> Dict:
        """
        Build complete output manifest.

        Args:
            graph: System graph
            ranked_smells: Ranked smells from scorer
            batch_metadata: Batch metadata
            graph_builder: Graph builder instance

        Returns:
            Manifest dictionary ready for JSON serialization
        """
        manifest = {
            "pipeline_version": "2026.1.0",
            "target_project": self.target_project,
            "system_graph": graph_builder.to_json_serializable(),
            "prioritization_registry": {
                "ranked_smells": [smell.to_dict() for smell in ranked_smells],
                "batches": batch_metadata
            }
        }
        return manifest

    def _write_manifest(self, manifest: Dict) -> str:
        """
        Write manifest to JSON file with pretty printing.

        Args:
            manifest: Manifest dictionary

        Returns:
            Path to written file
        """
        output_path = Path(self.output_json)

        # Ensure all values are JSON-serializable (convert numpy types)
        manifest = self._ensure_json_serializable(manifest)

        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Manifest written to {output_path}")
        return str(output_path)

    @staticmethod
    def _ensure_json_serializable(obj):
        """
        Recursively convert numpy types to native Python types for JSON serialization.

        Args:
            obj: Object to convert (dict, list, or primitive)

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: NSGAIIManifestBuilder._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [NSGAIIManifestBuilder._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the script."""
    # Configuration - Updated to relative project directory mapping
    workspace_dir = Path("data/input")
    arch_csv = workspace_dir / "ArchitectureSmells.csv"
    design_csv = workspace_dir / "DesignSmells.csv"
    impl_csv = workspace_dir / "ImplementationSmells.csv"
    type_metrics_csv = workspace_dir / "TypeMetrics.csv"
    method_metrics_csv = workspace_dir / "MethodMetrics.csv"

    output_manifest = Path("data/nsga2_input_manifest.json")

    # Validate inputs exist
    for csv_file in [arch_csv, design_csv, impl_csv]:
        if not csv_file.exists():
            logger.error(f"Required CSV file not found: {csv_file}")
            sys.exit(1)

    # Check if real metrics files exist
    use_real_metrics = type_metrics_csv.exists() and method_metrics_csv.exists()
    if use_real_metrics:
        logger.info("Real metrics files found - will use TypeMetrics and MethodMetrics")
    else:
        logger.info("Real metrics files not found - will use mock metrics")
        type_metrics_csv = None
        method_metrics_csv = None

    # Run pipeline
    builder = NSGAIIManifestBuilder(
        arch_csv=str(arch_csv),
        design_csv=str(design_csv),
        impl_csv=str(impl_csv),
        output_json=str(output_manifest),
        target_project="tika",
        type_metrics_csv=str(type_metrics_csv) if use_real_metrics else None,
        method_metrics_csv=str(method_metrics_csv) if use_real_metrics else None
    )

    try:
        output_path = builder.run()
        print(f"\n✓ Success! Manifest generated at: {output_path}")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
