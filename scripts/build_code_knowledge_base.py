#!/usr/bin/env python3
"""Build a lightweight code knowledge graph from Python and git history."""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from hybrid_rag.code_graph_builder import build_graph

SKIP_PARTS = {".git", ".venv", ".phoenix", ".chroma", "__pycache__"}


def module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = dotted(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return None


def python_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.py")
        if not any(part in SKIP_PARTS or part.startswith((".phoenix", ".venv", ".chroma")) for part in p.parts)
    )


def git_cochange(root: Path) -> Counter[tuple[str, str]]:
    try:
        text = subprocess.check_output(
            ["git", "log", "--format=commit:%H", "--name-only", "--diff-filter=ACMR"],
            cwd=root, text=True, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return Counter()
    counts: Counter[tuple[str, str]] = Counter()
    for block in text.split("commit:")[1:]:
        files = sorted({line.strip().replace("\\", "/") for line in block.splitlines()[1:] if line.strip()})
        # Ignore downloaded laws, generated indexes, and media. Co-change
        # edges are intended to describe the maintainable codebase, not data
        # snapshots that can create millions of noisy pairs.
        files = [
            path for path in files
            if Path(path).suffix.casefold() in {".py", ".md", ".toml", ".yaml", ".yml", ".json"}
            and not path.startswith(("data/", "outputs/", "logs/"))
        ]
        for left, right in itertools.combinations(files, 2):
            counts[(left, right)] += 1
    return counts


def build(root: Path) -> dict:
    return build_graph(root)
    # Legacy implementation retained below temporarily for compatibility with
    # older imports; the v2 builder above is the active implementation.
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    edge_seen: set[tuple[str, str, str]] = set()
    functions: dict[str, list[str]] = defaultdict(list)
    imports_by_module: dict[str, str] = {}
    module_by_leaf: dict[str, str] = {}

    def node(node_id: str, kind: str, **data: object) -> None:
        nodes.setdefault(node_id, {"id": node_id, "kind": kind, **data})

    files = python_files(root)
    for path in files:
        mod = module_name(path, root)
        module_by_leaf.setdefault(mod.rsplit(".", 1)[-1], mod)

    for path in files:
        rel = path.relative_to(root).as_posix()
        mod = module_name(path, root)
        file_id = f"file:{rel}"
        node(file_id, "file", path=rel, module=mod)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for item in ast.walk(tree):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{mod}.{item.name}"
                fid = f"function:{qual}"
                node(fid, "function", name=item.name, module=mod, file=rel, line=item.lineno)
                functions[item.name].append(fid)
            elif isinstance(item, ast.Import):
                for alias in item.names:
                    name = alias.name
                    iid = f"import:{mod}:{name}"
                    node(iid, "import", name=name, module=mod, file=rel, line=item.lineno)
                    imports_by_module[name] = iid
            elif isinstance(item, ast.ImportFrom):
                base = "." * item.level + (item.module or "")
                for alias in item.names:
                    name = f"{base}:{alias.name}"
                    iid = f"import:{mod}:{name}"
                    node(iid, "import", name=name, module=mod, file=rel, line=item.lineno)
                    imports_by_module[f"{item.module or ''}.{alias.name}"] = iid

        # Import declaration -> imported declaration (repo-local when resolvable).
        for item in ast.walk(tree):
            if isinstance(item, ast.Import):
                for alias in item.names:
                    source = f"import:{mod}:{alias.name}"
                    target_mod = alias.name if alias.name in module_by_leaf.values() else module_by_leaf.get(alias.name.rsplit(".", 1)[-1])
                    target = f"import:{target_mod}:module" if target_mod else imports_by_module.get(alias.name)
                    if target_mod:
                        node(target, "import", name=target_mod, module=target_mod)
                    if target and target != source:
                        edges.append({"type": "import_to_import", "source": source, "target": target, "file": rel})
            elif isinstance(item, ast.ImportFrom):
                base = item.module or ""
                for alias in item.names:
                    source = f"import:{mod}:{'.' * item.level + base}:{alias.name}"
                    target_mod = module_by_leaf.get((base or alias.name).rsplit(".", 1)[-1])
                    target = f"import:{target_mod}:module" if target_mod else imports_by_module.get(f"{base}.{alias.name}")
                    if target_mod:
                        node(target, "import", name=target_mod, module=target_mod)
                    if target and target != source:
                        edges.append({"type": "import_to_import", "source": source, "target": target, "file": rel})

        # Function call edges, resolved against local function names.
        for owner in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            source = f"function:{mod}.{owner.name}"
            for call in [n for n in ast.walk(owner) if isinstance(n, ast.Call)]:
                called = dotted(call.func)
                if not called:
                    continue
                name = called.rsplit(".", 1)[-1]
                targets = functions.get(name, [])
                for target in targets:
                    if target != source:
                        edges.append({"type": "function_to_function", "source": source, "target": target, "call": called, "file": rel})

    # Co-edited file pairs from git history.
    for (left, right), weight in git_cochange(root).items():
        left_id, right_id = f"file:{left}", f"file:{right}"
        node(left_id, "file", path=left)
        node(right_id, "file", path=right)
        edges.append({"type": "file_to_file", "source": left_id, "target": right_id, "cochange_count": weight})

    # Deduplicate AST edges while retaining evidence fields.
    unique: dict[tuple[str, str, str], dict] = {}
    for edge in edges:
        key = (edge["type"], edge["source"], edge["target"])
        unique.setdefault(key, edge)
    edges = list(unique.values())
    counts = Counter(edge["type"] for edge in edges)
    return {
        "schema_version": 1,
        "generated_from": str(root),
        "nodes": list(nodes.values()),
        "edges": edges,
        "stats": {"nodes": len(nodes), "edges": len(edges), **counts},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/code_knowledge_base.json"))
    args = parser.parse_args()
    graph = build(args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(json.dumps(graph["stats"], indent=2))


if __name__ == "__main__":
    main()
