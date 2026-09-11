#!/usr/bin/env python
"""Read-only source inventory. Unreferenced entrypoints are NOT deletion proof."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

SKIP = {'.git', '.codex', '.agents', '.claude', '.venv', 'venv', 'node_modules',
        '__pycache__', 'outputs', 'checkpoints', 'archive', 'logs', 'dist', 'build'}
SOURCE_SUFFIXES = {'.py', '.sh', '.json', '.yaml', '.yml', '.toml', '.md', '.txt', '.js', '.cjs'}


def source_files(root):
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix in SOURCE_SUFFIXES:
                yield path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_pins(root):
    pins = defaultdict(list)
    for directory, dirs, files in os.walk(root / 'outputs'):
        dirs[:] = [d for d in dirs if d not in {'cache', 'arrays', 'data', 'code_snapshot', 'weights', 'frames', '__pycache__'}]
        if 'run_provenance.json' not in files:
            continue
        path = Path(directory) / 'run_provenance.json'
        try:
            values = json.loads(path.read_text()).get('sha256', {})
        except (ValueError, OSError):
            continue
        if not isinstance(values, dict):
            continue
        for name, digest in values.items():
            target = Path(name)
            if target.is_relative_to(root) and 'outputs' not in target.relative_to(root).parts:
                pins[str(target.relative_to(root))].append({'manifest': str(path.relative_to(root)), 'sha256': digest})
    return dict(pins)


def audit(root):
    paths = list(source_files(root))
    modules = {'.'.join(p.relative_to(root).with_suffix('').parts).removesuffix('.__init__'): str(p.relative_to(root))
               for p in paths if p.suffix == '.py'}
    rows, parse_errors = [], []
    imported_by, full_duplicates, function_duplicates = defaultdict(set), defaultdict(list), defaultdict(list)
    for path in paths:
        if path.suffix != '.py':
            continue
        name = str(path.relative_to(root))
        text = path.read_text(errors='replace')
        full_duplicates[hashlib.sha256(text.encode()).hexdigest()].append(name)
        try:
            tree = ast.parse(text, filename=name)
        except SyntaxError as error:
            parse_errors.append({'path': name, 'line': error.lineno, 'message': error.msg})
            continue
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        imports = set()
        for node in ast.walk(tree):
            candidates = []
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                package = name.removesuffix('.py').replace('/', '.').split('.')[:-1]
                if node.level:
                    package = package[:len(package)-node.level+1]
                    base = '.'.join(package + ([node.module] if node.module else []))
                else:
                    base = node.module or ''
                candidates = [base] + [base + '.' + alias.name for alias in node.names]
            for candidate in candidates:
                while candidate:
                    if candidate in modules:
                        imports.add(modules[candidate]); imported_by[modules[candidate]].add(name)
                        break
                    candidate = candidate.rpartition('.')[0]
        for node in definitions:
            size = node.end_lineno - node.lineno + 1
            if not isinstance(node, ast.ClassDef) and size >= 8:
                digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                function_duplicates[digest].append({'path':name, 'name':node.name, 'line':node.lineno, 'lines':size})
        rows.append({'path':name, 'lines':len(text.splitlines()), 'sha256':sha256(path),
                     'imports_local':sorted(imports),
                     'definitions':[{'name':n.name,'line':n.lineno,'lines':n.end_lineno-n.lineno+1,'kind':type(n).__name__} for n in definitions],
                     'has_main_guard':any(isinstance(n, ast.If) and '__name__' in ast.unparse(n.test) and '__main__' in ast.unparse(n.test) for n in tree.body)})
    for row in rows:
        row['imported_by'] = sorted(imported_by[row['path']])
    return {
        'scope':'Static Python audit excluding runtime assets, archives, environments and vendored node_modules. Imports include conditional imports; absence of imports is NOT proof of unused code.',
        'python_files':len(rows), 'python_lines':sum(r['lines'] for r in rows),
        'areas':dict(Counter(r['path'].split('/')[0] if '/' in r['path'] else '(root)' for r in rows)),
        'files':sorted(rows,key=lambda r:r['path']), 'parse_errors':parse_errors,
        'identical_file_groups':[v for v in full_duplicates.values() if len(v)>1],
        'identical_top_level_function_groups':[v for v in function_duplicates.values() if len(v)>1],
        'experiment_pins':experiment_pins(root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['python_files','python_lines','areas','parse_errors']}))
    print('Exact duplicate file groups:',len(result['identical_file_groups']))
    print('Exact duplicate function groups:',len(result['identical_top_level_function_groups']))

if __name__ == '__main__':
    main()
