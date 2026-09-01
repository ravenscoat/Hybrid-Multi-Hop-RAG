from __future__ import annotations

import ast
import itertools
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SKIP_PARTS = {
    ".git", ".venv", ".phoenix", ".chroma", "__pycache__", ".pytest_cache",
    "data", "documents", "markdown", "markdowns", "outputs", "logs",
}
MAINTAINABLE_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".json"}


def _module_name(path: Path, root: Path) -> str:
    return ".".join(path.relative_to(root).with_suffix("").parts)


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _dotted(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return None


def _python_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for current, directories, filenames in os.walk(root):
        directories[:] = [name for name in directories if name not in SKIP_PARTS and not name.startswith((".phoenix", ".chroma", ".venv"))]
        result.extend(Path(current) / name for name in filenames if name.endswith(".py"))
    return sorted(result)


def _resolve_relative_module(current: str, level: int, imported: str) -> str:
    if not level:
        return imported
    package = current.rpartition(".")[0].split(".")
    prefix = package[: max(0, len(package) - level + 1)]
    return ".".join([*prefix, *([imported] if imported else [])])


def _function_records(
    tree: ast.Module, module: str
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, str | None]]:
    records: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, str | None]] = []
    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            records.append((f"{module}.{item.name}", item, None))
        elif isinstance(item, ast.ClassDef):
            for child in item.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    records.append((f"{module}.{item.name}.{child.name}", child, item.name))
    return records


def _git_cochanges(root: Path) -> tuple[Counter[tuple[str, str]], bool]:
    try:
        text = subprocess.check_output(
            [
                "git", "log", "--format=commit:%H|%ct", "--name-only", "--diff-filter=ACMR", "--", ".",
                ":(exclude)data/**", ":(exclude)outputs/**", ":(exclude)logs/**",
                ":(exclude).chroma/**", ":(exclude).phoenix*/**", ":(exclude).venv/**",
            ],
            cwd=root,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return Counter(), False
    now = datetime.now(timezone.utc).timestamp()
    scores: Counter[tuple[str, str]] = Counter()
    for block in text.split("commit:")[1:]:
        lines = block.splitlines()
        try:
            timestamp = int(lines[0].split("|", 1)[1])
        except (IndexError, ValueError):
            timestamp = int(now)
        age_years = max(0.0, (now - timestamp) / (365.25 * 86400))
        recency = max(0.25, 0.5 ** (age_years / 2.0))
        paths = sorted({line.strip().replace("\\", "/") for line in lines[1:] if line.strip()})
        paths = [
            path for path in paths
            if Path(path).suffix.casefold() in MAINTAINABLE_SUFFIXES
            and not any(part in SKIP_PARTS for part in Path(path).parts)
        ]
        for left, right in itertools.combinations(paths, 2):
            scores[(left, right)] += recency
    return scores, True


def build_graph(root: Path) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    functions_by_leaf: dict[str, list[str]] = defaultdict(list)
    functions_by_qual: dict[str, str] = {}
    modules: dict[str, str] = {}
    parsed: list[tuple[str, str, str, ast.Module, list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, str | None]]]] = []
    diagnostics: dict[str, Any] = {"parse_failures": [], "ambiguous_calls": 0, "unresolved_calls": 0}

    def add_node(node_id: str, kind: str, **attrs: Any) -> None:
        current = nodes.setdefault(node_id, {"id": node_id, "kind": kind})
        for key, value in attrs.items():
            if value not in (None, ""):
                current.setdefault(key, value)

    for path in _python_files(root):
        rel, module = path.relative_to(root).as_posix(), _module_name(path, root)
        modules[module] = rel
        source = path.read_text(encoding="utf-8", errors="replace")
        file_id = f"file:{rel}"
        add_node(file_id, "file", path=rel, module=module, semantic_text=f"Python file {rel}. Module {module}.")
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            diagnostics["parse_failures"].append({"file": rel, "line": exc.lineno, "error": exc.msg})
            continue
        records = _function_records(tree, module)
        parsed.append((rel, module, source, tree, records))
        for qual, item, owner_class in records:
            fid = f"function:{qual}"
            try:
                signature = ast.unparse(item.args)
            except Exception:
                signature = ""
            docstring = ast.get_docstring(item) or ""
            excerpt = (ast.get_source_segment(source, item) or "")[:700]
            add_node(
                fid,
                "function",
                name=item.name,
                qualified_name=qual,
                module=module,
                file=rel,
                line=item.lineno,
                class_name=owner_class,
                signature=signature,
                docstring=docstring,
                source_excerpt=excerpt,
                semantic_text=f"Function {qual} in {rel}. Signature {signature}. {docstring}. {excerpt}",
            )
            functions_by_leaf[item.name].append(fid)
            functions_by_qual[qual] = fid
            edges.extend([
                {"type": "file_contains_function", "source": file_id, "target": fid, "weight": 2.0},
                {"type": "function_defined_in_file", "source": fid, "target": file_id, "weight": 2.0},
            ])

    for rel, module, _, tree, records in parsed:
        file_id = f"file:{rel}"
        aliases: dict[str, str] = {}
        for item in ast.walk(tree):
            imported_items: list[tuple[str, str, str]] = []
            if isinstance(item, ast.Import):
                imported_items = [(alias.asname or alias.name.split(".")[0], alias.name, alias.name) for alias in item.names]
            elif isinstance(item, ast.ImportFrom):
                base = _resolve_relative_module(module, item.level, item.module or "")
                imported_items = [(alias.asname or alias.name, f"{base}.{alias.name}".strip("."), base) for alias in item.names]
            for local_name, imported_name, target_module in imported_items:
                iid = f"import:{module}:{imported_name}"
                target_file = modules.get(target_module)
                if not target_file:
                    candidates = [name for name in modules if name == target_module or name.endswith(f".{target_module}")]
                    if len(candidates) == 1:
                        target_module, target_file = candidates[0], modules[candidates[0]]
                add_node(iid, "import", name=imported_name, local_name=local_name, module=module, file=rel, line=item.lineno, semantic_text=f"Import {imported_name} as {local_name} in {rel}")
                edges.append({"type": "file_contains_import", "source": file_id, "target": iid, "weight": 1.25})
                aliases[local_name] = imported_name
                if target_file:
                    target_import = f"import:{target_module}:module"
                    add_node(target_import, "import", name=target_module, module=target_module, file=target_file, semantic_text=f"Repository module {target_module} in {target_file}")
                    edges.extend([
                        {"type": "import_to_import", "source": iid, "target": target_import, "file": rel, "line": item.lineno, "confidence": 1.0},
                        {"type": "import_to_file", "source": iid, "target": f"file:{target_file}", "file": rel, "line": item.lineno, "confidence": 1.0, "bidirectional": True},
                    ])

        for qual, owner, owner_class in records:
            source_id = f"function:{qual}"
            for call in (node for node in ast.walk(owner) if isinstance(node, ast.Call)):
                called = _dotted(call.func)
                if not called:
                    continue
                leaf = called.rsplit(".", 1)[-1]
                target: str | None = None
                if called.startswith("self.") and owner_class:
                    target = functions_by_qual.get(f"{module}.{owner_class}.{leaf}")
                elif "." not in called:
                    target = functions_by_qual.get(f"{module}.{called}") or functions_by_qual.get(aliases.get(called, ""))
                else:
                    head, tail = called.split(".", 1)
                    expanded = f"{aliases.get(head, head)}.{tail}"
                    target = functions_by_qual.get(expanded)
                    if not target:
                        suffixes = [fid for name, fid in functions_by_qual.items() if name.endswith(f".{expanded}")]
                        target = suffixes[0] if len(suffixes) == 1 else None
                if not target:
                    fallback = functions_by_leaf.get(leaf, [])
                    if len(fallback) == 1:
                        target = fallback[0]
                    elif len(fallback) > 1:
                        diagnostics["ambiguous_calls"] += 1
                    else:
                        diagnostics["unresolved_calls"] += 1
                if target and target != source_id:
                    edges.append({"type": "function_to_function", "source": source_id, "target": target, "call": called, "file": rel, "line": call.lineno, "confidence": 1.0, "weight": 1.5})

    cochanges, git_available = _git_cochanges(root)
    diagnostics["git_history_available"] = git_available
    for (left, right), weighted_count in cochanges.items():
        left_id, right_id = f"file:{left}", f"file:{right}"
        add_node(left_id, "file", path=left, semantic_text=f"Repository file {left}")
        add_node(right_id, "file", path=right, semantic_text=f"Repository file {right}")
        edges.append({"type": "file_to_file", "source": left_id, "target": right_id, "cochange_count": round(weighted_count, 4), "bidirectional": True})

    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        key = edge["type"], edge["source"], edge["target"]
        current = unique.get(key)
        if current is None or edge.get("cochange_count", 0) > current.get("cochange_count", 0):
            unique[key] = edge
    final_edges = list(unique.values())
    return {
        "schema_version": 2,
        "generated_from": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": list(nodes.values()),
        "edges": final_edges,
        "stats": {
            "nodes": len(nodes),
            "edges": len(final_edges),
            "node_types": dict(Counter(node["kind"] for node in nodes.values())),
            "edge_types": dict(Counter(edge["type"] for edge in final_edges)),
        },
        "diagnostics": diagnostics,
    }
