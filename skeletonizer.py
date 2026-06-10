
import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import javalang
import networkx as nx



# CONFIGURATION
JAVA_EXT = ".java"

IGNORED_DIRS = {
    ".git",
    ".idea",
    "target",
    "build",
    "node_modules",
    ".gradle",
    ".mvn"
}

MAX_METHOD_BODY_STUB = "/* Body Pruned */"


# REGEX CLEANUP UTILITIES # Will add more later

def remove_comments(source: str) -> str:
    """
    Removes Java comments using regex.
    """
    pattern = r"""
        (//.*?$)                         | # single-line comments
        (/\*.*?\*/)                       # multi-line comments
    """
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL | re.VERBOSE)
    return regex.sub("", source)


def normalize_whitespace(source: str) -> str:
    """
    Normalize excessive whitespace.
    """
    source = re.sub(r"\n\s*\n\s*\n+", "\n\n", source)
    return source.strip()



# FILE SYSTEM EXPLORER

class RepositoryExplorer:

    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)

    def discover_java_files(self):
        java_files = []

        for root, dirs, files in os.walk(self.repo_path):

            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

            for file in files:
                if file.endswith(JAVA_EXT):
                    java_files.append(Path(root) / file)

        return java_files



# JAVA SKELETONIZER

class JavaSkeletonizer:

    def __init__(self):
        self.graph = nx.DiGraph()

        self.context_map = []
        self.metrics = {}

        self.parsing_failures = []

    
    # MAIN FILE PROCESSOR   
    def process_repository(self, repo_path, output_dir):

        repo_path = Path(repo_path)
        output_dir = Path(output_dir)

        skeleton_dir = output_dir / "skeletons"

        skeleton_dir.mkdir(parents=True, exist_ok=True)

        explorer = RepositoryExplorer(repo_path)
        java_files = explorer.discover_java_files()

        print(f"[INFO] Found {len(java_files)} Java files.")

        total_original_chars = 0
        total_skeleton_chars = 0

        for java_file in java_files:

            try:
                with open(java_file, "r", encoding="utf-8", errors="ignore") as f:
                    raw_source = f.read()

                total_original_chars += len(raw_source)

                cleaned_source = remove_comments(raw_source)

                skeleton, context_data = self.skeletonize_file(
                    cleaned_source,
                    java_file
                )

                total_skeleton_chars += len(skeleton)

                relative_path = java_file.relative_to(repo_path)

                output_file = skeleton_dir / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with open(output_file, "w", encoding="utf-8") as out:
                    out.write(skeleton)

                self.context_map.append(context_data)

            except Exception as e:
                self.parsing_failures.append({
                    "file": str(java_file),
                    "error": str(e)
                })

        # Build graph metrics
        self.compute_graph_metrics()

        # Export outputs
        self.export_outputs(output_dir)

        reduction = (
            1 - (total_skeleton_chars / max(total_original_chars, 1))
        ) * 100

        print("\n===================================================")
        print("SKELETONIZATION COMPLETE")
        print("===================================================")
        print(f"Original chars : {total_original_chars}")
        print(f"Skeleton chars : {total_skeleton_chars}")
        print(f"Reduction       : {reduction:.2f}%")
        print(f"Nodes           : {self.graph.number_of_nodes()}")
        print(f"Edges           : {self.graph.number_of_edges()}")
        print("===================================================")

    
    # FILE SKELETONIZATION
    def skeletonize_file(self, source_code, file_path):

        try:
            tree = javalang.parse.parse(source_code)

        except Exception as e:
            return (
                f"// PARSE FAILURE: {e}",
                {
                    "file": str(file_path),
                    "parse_failure": True
                }
            )

        skeleton_lines = []

        package_name = ""
        imports = []
        types = []

        # PACKAGE
        if tree.package:
            package_name = tree.package.name
            skeleton_lines.append(f"package {package_name};")
            skeleton_lines.append("")

        
        # IMPORTS
        for imp in tree.imports:

            wildcard = ".*" if imp.wildcard else ""

            import_stmt = f"import {imp.path}{wildcard};"

            imports.append(imp.path)

            skeleton_lines.append(import_stmt)

        skeleton_lines.append("")

        # TYPE DECLARATIONS
        for _, node in tree.filter(javalang.tree.TypeDeclaration):

            type_data = self.extract_type(node)

            types.append(type_data)

            skeleton_lines.extend(
                self.render_type_skeleton(type_data)
            )

            self.add_type_dependencies(type_data)

        skeleton = "\n".join(skeleton_lines)
        skeleton = normalize_whitespace(skeleton)

        context_data = {
            "file": str(file_path),
            "package": package_name,
            "imports": imports,
            "types": types
        }

        return skeleton, context_data

    
    # TYPE EXTRACTION
    def extract_type(self, node):

        type_kind = node.__class__.__name__

        modifiers = list(node.modifiers) if node.modifiers else []

        extends = None
        implements = []

        if hasattr(node, "extends") and node.extends:
            extends = node.extends.name

        if hasattr(node, "implements") and node.implements:
            implements = [i.name for i in node.implements]

        annotations = []
        if hasattr(node, "annotations"):
            annotations = [
                a.name for a in node.annotations
            ]

        fields = self.extract_fields(node)
        methods = self.extract_methods(node)

        return {
            "kind": type_kind,
            "name": node.name,
            "modifiers": modifiers,
            "annotations": annotations,
            "extends": extends,
            "implements": implements,
            "fields": fields,
            "methods": methods
        }

    
    # FIELD EXTRACTION
    def extract_fields(self, node):

        fields = []

        if not hasattr(node, "fields"):
            return fields

        for field in node.fields:

            field_type = self.safe_type_name(field.type)

            field_modifiers = (
                list(field.modifiers)
                if field.modifiers else []
            )

            for declarator in field.declarators:

                fields.append({
                    "name": declarator.name,
                    "type": field_type,
                    "modifiers": field_modifiers
                })

        return fields

    
    # METHOD EXTRACTION
    def extract_methods(self, node):

        methods = []

        if not hasattr(node, "methods"):
            return methods

        for method in node.methods:

            return_type = (
                self.safe_type_name(method.return_type)
                if method.return_type else "void"
            )

            params = []

            for p in method.parameters:

                params.append({
                    "name": p.name,
                    "type": self.safe_type_name(p.type)
                })

            throws = []

            if method.throws:
                throws = [t for t in method.throws]

            annotations = []
            if method.annotations:
                annotations = [a.name for a in method.annotations]

            methods.append({
                "name": method.name,
                "return_type": return_type,
                "modifiers": list(method.modifiers),
                "parameters": params,
                "throws": throws,
                "annotations": annotations
            })

        return methods

    
    # TYPE NAME SAFETY
    def safe_type_name(self, type_node):

        if type_node is None:
            return "Unknown"

        try:
            if hasattr(type_node, "name"):
                return type_node.name

            return str(type_node)

        except Exception:
            return "Unknown"

    
    # RENDER SKELETON
    def render_type_skeleton(self, type_data):

        lines = []

        annotations = type_data["annotations"]
        modifiers = " ".join(type_data["modifiers"])

        kind = type_data["kind"].replace("Declaration", "").lower()

        name = type_data["name"]

        extends_clause = (
            f" extends {type_data['extends']}"
            if type_data["extends"]
            else ""
        )

        implements_clause = ""

        if type_data["implements"]:
            implements_clause = (
                " implements "
                + ", ".join(type_data["implements"])
            )

        for ann in annotations:
            lines.append(f"@{ann}")

        lines.append(
            f"{modifiers} {kind} {name}"
            f"{extends_clause}"
            f"{implements_clause} {{"
        )

        # Fields
        for field in type_data["fields"]:

            field_mods = " ".join(field["modifiers"])

            lines.append(
                f"    {field_mods} "
                f"{field['type']} "
                f"{field['name']};"
            )

        if type_data["fields"]:
            lines.append("")

        # Methods
        for method in type_data["methods"]:

            for ann in method["annotations"]:
                lines.append(f"    @{ann}")

            method_mods = " ".join(method["modifiers"])

            params = ", ".join([
                f"{p['type']} {p['name']}"
                for p in method["parameters"]
            ])

            throws_clause = ""

            if method["throws"]:
                throws_clause = (
                    " throws "
                    + ", ".join(method["throws"])
                )

            lines.append(
                f"    {method_mods} "
                f"{method['return_type']} "
                f"{method['name']}({params})"
                f"{throws_clause} "
                f"{{ {MAX_METHOD_BODY_STUB} }}"
            )

        lines.append("}\n")

        return lines

    
    # DEPENDENCY GRAPH
    def add_type_dependencies(self, type_data):

        source = type_data["name"]

        self.graph.add_node(source)

        # Extends
        if type_data["extends"]:
            self.graph.add_edge(
                source,
                type_data["extends"],
                relation="extends"
            )

        # Implements
        for impl in type_data["implements"]:

            self.graph.add_edge(
                source,
                impl,
                relation="implements"
            )

        # Fields
        for field in type_data["fields"]:

            self.graph.add_edge(
                source,
                field["type"],
                relation="field_dependency"
            )

        # Method params + returns
        for method in type_data["methods"]:

            self.graph.add_edge(
                source,
                method["return_type"],
                relation="returns"
            )

            for param in method["parameters"]:

                self.graph.add_edge(
                    source,
                    param["type"],
                    relation="parameter"
                )

    
    # GRAPH METRICS
    def compute_graph_metrics(self):

        print("[INFO] Computing graph metrics...")

        metrics = {}

        try:
            metrics["degree_centrality"] = nx.degree_centrality(
                self.graph
            )

            metrics["betweenness_centrality"] = (
                nx.betweenness_centrality(self.graph)
            )

            metrics["pagerank"] = nx.pagerank(self.graph)

            metrics["strongly_connected_components"] = [
                list(c)
                for c in nx.strongly_connected_components(self.graph)
                if len(c) > 1
            ]

        except Exception as e:
            print(f"[WARNING] Metric computation failed: {e}")

        self.metrics = metrics

    
    # EXPORTS
    def export_outputs(self, output_dir):

        output_dir = Path(output_dir)

        # Context map
        with open(
            output_dir / "context_map.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(self.context_map, f, indent=2)

        # Metrics
        with open(
            output_dir / "metrics.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(self.metrics, f, indent=2)

        # Parsing failures
        with open(
            output_dir / "parsing_failures.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(self.parsing_failures, f, indent=2)

        # Important classes
        important = self.rank_important_classes()

        with open(
            output_dir / "important_classes.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(important, f, indent=2)

        # Export graph
        try:
            nx.write_graphml(
                self.graph,
                output_dir / "dependency_graph.graphml"
            )
        except Exception as e:
            print(f"[WARNING] Graph export failed: {e}")

    
    # IMPORTANT CLASS RANKING
    def rank_important_classes(self):

        pagerank = self.metrics.get("pagerank", {})

        ranked = sorted(
            pagerank.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {
                "class": cls,
                "importance": score
            }
            for cls, score in ranked[:100]
        ]



# # MAin
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "repo",
        help="Path to Java repository"
    )

    parser.add_argument(
        "output",
        help="Output directory"
    )

    args = parser.parse_args()

    skeletonizer = JavaSkeletonizer()

    skeletonizer.process_repository(
        args.repo,
        args.output
    )


if __name__ == "__main__":
    main()