import pathlib
from typing import List, Tuple

ROOT = pathlib.Path(__file__).resolve().parent
INCLUDE_DIRS = {
    'algorithms', 'config', 'core', 'data', 'experiments', 'pandas', 'pydantic', 'problems',
    'reporting', 'simulation', 'utils', 'validation', 'visualization', 'scripts', 'yaml'
}
EXTRA_FILES = ['rms_all_in_one.py']
EXCLUDED = {'tests'}
RESOURCE_PATHS = [
    pathlib.Path('config/base_config.yaml'),
    pathlib.Path('data/benchmarks/fisher_jsp_6x6.csv'),
    pathlib.Path('data/benchmarks/industry_case_cell.csv'),
    pathlib.Path('data/benchmarks/taillard_fsp_5x5.csv'),
    pathlib.Path('data/synthetic/sample.csv'),
]


def gather_sources() -> List[Tuple[str, bool, str, pathlib.Path]]:
    entries = []
    for path in ROOT.rglob('*.py'):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in EXCLUDED:
            continue
        if rel.name == 'rms_monolith_full.py':
            continue
        if rel.parts[0] not in INCLUDE_DIRS and str(rel) not in EXTRA_FILES:
            continue
        module_parts = list(rel.parts)
        module_parts[-1] = module_parts[-1][:-3]
        is_package = False
        if module_parts[-1] == '__init__':
            module_parts = module_parts[:-1]
            is_package = True
        module_name = '.'.join(module_parts) if module_parts else path.stem
        with path.open('r', encoding='utf-8') as handle:
            source = handle.read().rstrip() + '\n'
        entries.append((module_name, is_package, source, rel))
    seen = set()
    unique = []
    for entry in entries:
        name = entry[0]
        if name in seen:
            continue
        seen.add(name)
        unique.append(entry)
    unique.sort(key=lambda item: (item[0].count('.'), item[0]))
    return unique


def gather_resources() -> List[Tuple[str, str]]:
    resources = []
    for rel in RESOURCE_PATHS:
        path = ROOT / rel
        if not path.exists():
            continue
        content = path.read_text(encoding='utf-8')
        resources.append((str(rel).replace('\\', '/'), content))
    return resources


def encode(source: str) -> str:
    if "'''" not in source:
        return "r'''\n" + source + "'''"
    if '"""' not in source:
        return 'r"""\n' + source + '"""'
    escaped = source.replace("'''", "\\'\\'\\'")
    return "r'''\n" + escaped + "'''"


def build() -> None:
    modules = gather_sources()
    resources = gather_resources()
    out_path = ROOT / 'rms_monolith_full.py'
    with out_path.open('w', encoding='utf-8') as out:
        out.write('# Auto-generated monolithic RMS optimisation script\n')
        out.write('# Do not edit manually.\n\n')
        out.write('"""Monolithic RMS optimisation framework."""\n\n')
        out.write('from __future__ import annotations\n\n')
        out.write('import os\nimport sys\nimport types\nfrom pathlib import Path\n\n')
        out.write('MODULE_SOURCES: dict[str, dict[str, object]] = {}\n\n')
        out.write('def _register_module(name: str, source: str, is_package: bool) -> None:\n')
        out.write('    MODULE_SOURCES[name] = {"code": source, "is_package": is_package}\n\n')
        for name, is_package, source, rel in modules:
            out.write(f"# BEGIN MODULE: {name} ({rel})\n")
            out.write(f"_register_module('{name}', {encode(source)}, {is_package})\n")
            out.write(f"# END MODULE: {name}\n\n")
        out.write('RESOURCE_FILES: dict[str, str] = {}\n\n')
        out.write('def _register_resource(path: str, content: str) -> None:\n')
        out.write('    RESOURCE_FILES[path] = content\n\n')
        for rel_path, content in resources:
            out.write(f"# BEGIN RESOURCE: {rel_path}\n")
            out.write(f"_register_resource('{rel_path}', {encode(content)})\n")
            out.write(f"# END RESOURCE: {rel_path}\n\n")
        out.write('def bootstrap_environment(base_path: Path | None = None) -> Path:\n')
        out.write('    if base_path is None:\n')
        out.write('        base_path = Path.cwd() / "rms_runtime"\n')
        out.write('    base_path.mkdir(parents=True, exist_ok=True)\n')
        out.write('    for name, meta in MODULE_SOURCES.items():\n')
        out.write('        module_path = base_path / Path(name.replace(".", os.sep))\n')
        out.write('        if meta["is_package"]:\n')
        out.write('            module_path.mkdir(parents=True, exist_ok=True)\n')
        out.write('            init_file = module_path / "__init__.py"\n')
        out.write('            init_file.write_text(meta["code"], encoding="utf-8")\n')
        out.write('        else:\n')
        out.write('            module_path.parent.mkdir(parents=True, exist_ok=True)\n')
        out.write('            py_file = module_path.with_suffix(".py")\n')
        out.write('            py_file.write_text(meta["code"], encoding="utf-8")\n')
        out.write('    for rel_path, content in RESOURCE_FILES.items():\n')
        out.write('        target = base_path / rel_path\n')
        out.write('        target.parent.mkdir(parents=True, exist_ok=True)\n')
        out.write('        target.write_text(content, encoding="utf-8")\n')
        out.write('    return base_path\n\n')
        out.write('def bootstrap_modules(base_path: Path | None = None) -> None:\n')
        out.write('    if base_path is None:\n')
        out.write('        base_path = Path.cwd() / "rms_runtime"\n')
        out.write('    if str(base_path) not in sys.path:\n')
        out.write('        sys.path.insert(0, str(base_path))\n')
        out.write('    for name, meta in sorted(MODULE_SOURCES.items(), key=lambda item: item[0].count(".")):\n')
        out.write('        module = types.ModuleType(name)\n')
        out.write('        module.__file__ = str((base_path / Path(name.replace(".", os.sep))).with_suffix(".py"))\n')
        out.write('        if meta["is_package"]:\n')
        out.write('            module.__path__ = [str(base_path / Path(name.replace(".", os.sep)))]\n')
        out.write('            module.__package__ = name\n')
        out.write('        else:\n')
        out.write('            module.__package__ = name.rsplit(".", 1)[0] if "." in name else ""\n')
        out.write('        sys.modules[name] = module\n')
        out.write('        exec(compile(meta["code"], module.__file__, "exec"), module.__dict__)\n\n')
        out.write('def main() -> None:\n')
        out.write('    base_path = bootstrap_environment()\n')
        out.write('    bootstrap_modules(base_path)\n')
        out.write('    from rms_all_in_one import main as orchestrator_main\n')
        out.write('    orchestrator_main()\n\n')
        out.write('if __name__ == "__main__":\n')
        out.write('    main()\n')


if __name__ == '__main__':
    build()
