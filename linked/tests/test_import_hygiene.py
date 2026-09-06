"""
Import-hygiene tests.

Guards against the packaging defect where a module imports a third-party
distribution at module level while ``install_requires`` never declares it.
Every environment the maintainers work in already has the package installed,
so the omission is invisible locally and only surfaces as an ``ImportError``
on a clean ``pip install linked``.

The check is AST-based (nothing is executed) and only looks at *unguarded
module-level* imports of the *runtime* package: an import nested in
``with suppress(ImportError):``, ``try``/``except ImportError``,
``if TYPE_CHECKING:`` or a function body is an optional dependency by
construction, and the test subpackage is skipped because its imports belong to
``extras_require``, not ``install_requires``.
"""

import ast
import configparser
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Dict, List, Set

import pytest

PACKAGE_NAME = "linked"
TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
REPO_DIR = PACKAGE_DIR.parent
SETUP_CFG = REPO_DIR / "setup.cfg"


def _declared_requirements(setup_cfg: Path) -> Set[str]:
    """Requirement names listed in ``[options] install_requires`` of *setup_cfg*."""
    config = configparser.ConfigParser()
    config.read(setup_cfg)
    raw = config.get("options", "install_requires", fallback="")
    requirements = set()
    for line in raw.splitlines():
        # Strip comments, extras and version specifiers: "foo[bar] >= 1.2  # note"
        name = line.split("#", 1)[0].strip()
        for separator in ("[", "=", "<", ">", "!", "~", ";", " "):
            name = name.split(separator, 1)[0]
        if name:
            requirements.add(name.strip())
    return requirements


def _importable_names(requirement: str) -> Set[str]:
    """Top-level import names that *requirement* provides.

    Falls back to the requirement's own (normalized) name when the
    distribution is not installed, so the check still works offline. When it
    *is* installed, its real top-level modules are used, which is what maps
    ``scikit-learn`` to ``sklearn``.
    """
    names = {requirement.replace("-", "_").lower()}
    try:
        dist = distribution(requirement)
    except PackageNotFoundError:
        return names
    top_level = dist.read_text("top_level.txt") or ""
    names.update(line.strip() for line in top_level.splitlines() if line.strip())
    for file in dist.files or ():
        parts = file.parts
        if len(parts) > 1 and parts[1] == "__init__.py":  # a top-level package
            names.add(parts[0])
        elif len(parts) == 1 and parts[0].endswith(".py"):  # a top-level module
            names.add(parts[0][: -len(".py")])
    return names


def _runtime_modules(package_dir: Path, *, excluding: Path) -> List[Path]:
    """The package's runtime ``.py`` modules, i.e. everything outside *excluding*."""
    return sorted(
        path for path in package_dir.rglob("*.py") if excluding not in path.parents
    )


def _unguarded_module_level_imports(module_path: Path) -> Set[str]:
    """Root module names imported unconditionally at the top level of *module_path*."""
    tree = ast.parse(module_path.read_text(), filename=str(module_path))
    roots = set()
    for node in tree.body:  # top level only: nested/guarded imports are opt-in
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


@pytest.mark.skipif(
    not SETUP_CFG.is_file(), reason="only meaningful in a source checkout"
)
def test_no_undeclared_module_level_imports():
    """Every unguarded module-level third-party import is in ``install_requires``."""
    provided = set()
    for requirement in _declared_requirements(SETUP_CFG):
        provided |= _importable_names(requirement)

    ignored = provided | set(sys.stdlib_module_names) | {PACKAGE_NAME}

    undeclared: Dict[str, List[str]] = {}
    for module_path in _runtime_modules(PACKAGE_DIR, excluding=TESTS_DIR):
        for root in _unguarded_module_level_imports(module_path) - ignored:
            undeclared.setdefault(root, []).append(
                str(module_path.relative_to(REPO_DIR))
            )

    assert not undeclared, (
        f"module-level imports missing from install_requires in {SETUP_CFG.name}: "
        f"{undeclared}"
    )
