"""
Import-hygiene guards: a clean install must be able to use the package.

A user's ``pip install`` acts on the declared dependencies and nothing else,
while a maintainer's environment always holds far more than that. A dependency
that is imported but never declared is therefore invisible locally and only
surfaces downstream -- as an ``ImportError``, or, when the import sits in a
``suppress(ImportError)`` block whose name is used anyway, as a much less
legible ``NameError``.

Rather than guess statically which imports are load-bearing (a guarded import
is *not* reliably optional -- see ``linked.cast``), these tests reproduce the
clean install: a subprocess imports the package with a ``sys.meta_path`` hook
that makes every distribution outside the declared dependency closure look
uninstalled. Whatever the package really needs at import time then fails
there, and only there.

Two invariants are checked:

- importing the package needs nothing beyond the declared dependencies;
- the declared dependencies are enough for *every* graph kind the package
  registers, so an undeclared distribution cannot silently drop a documented
  feature from clean installs.
"""

import json
import subprocess
import sys
from configparser import ConfigParser
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    packages_distributions,
)
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Set

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PACKAGE_NAME = "linked"
TESTS_DIR = Path(__file__).resolve().parent  # <repo>/<package>/tests
REPO_DIR = TESTS_DIR.parent.parent
ENCODING = "utf-8"

# Markers are evaluated for the running interpreter, with no extra requested:
# extras are opt-in, so they are never part of what a plain install provides.
NO_EXTRA = {"extra": ""}


# ---------------------------------------------------------------------------
# What the project declares
# ---------------------------------------------------------------------------


def _applicable_requirement_names(specifications: Iterable[str]) -> Set[str]:
    """Names of the *specifications* that apply to the running environment.

    Extras, version specifiers, comments and environment markers are stripped
    or resolved, so ``"foo[bar] >= 1.2 ; python_version < '3.9'  # note"``
    contributes ``foo`` on old interpreters and nothing on new ones.
    """
    names = set()
    for specification in specifications:
        text = specification.split("#", 1)[0].strip()
        if not text:
            continue
        requirement = Requirement(text)
        if requirement.marker is not None and not requirement.marker.evaluate(NO_EXTRA):
            continue
        names.add(requirement.name)
    return names


def _toml_parser():
    """The stdlib TOML reader, or the backport pytest brings in before 3.11."""
    for module_name in ("tomllib", "tomli"):
        with suppress(ModuleNotFoundError):
            return import_module(module_name)
    return None


def _pyproject_dependencies(path: Path) -> Optional[List[str]]:
    """``[project] dependencies`` of a ``pyproject.toml``, if it declares any."""
    toml = _toml_parser()
    if toml is None:  # pragma: no cover - only without tomllib and tomli
        pytest.fail(f"cannot read {path.name}: no TOML parser available")
    content = toml.loads(path.read_text(encoding=ENCODING))
    return content.get("project", {}).get("dependencies")


def _setup_cfg_install_requires(path: Path) -> Optional[List[str]]:
    """``[options] install_requires`` of a ``setup.cfg``, if it declares any."""
    config = ConfigParser()
    config.read(path, encoding=ENCODING)
    if not config.has_option("options", "install_requires"):
        return None
    return config.get("options", "install_requires").splitlines()


def _declared_requirements(repo_dir: Path) -> Optional[Set[str]]:
    """The project's runtime dependencies as declared in *repo_dir*.

    Returns ``None`` when *repo_dir* holds no packaging metadata at all, which
    is how the package looks when it runs from an installed distribution.
    Metadata that exists but declares nothing yields an empty set rather than
    ``None``: that is a claim about the project, and the probe below should
    hold the project to it instead of quietly skipping.

    ``pyproject.toml`` wins over ``setup.cfg`` because setuptools reads it that
    way, so the guard follows whichever metadata is actually in force -- and
    keeps working across the migration from one to the other.
    """
    readers = (
        (repo_dir / "pyproject.toml", _pyproject_dependencies),
        (repo_dir / "setup.cfg", _setup_cfg_install_requires),
    )
    metadata_found = False
    for path, read in readers:
        if not path.is_file():
            continue
        metadata_found = True
        declared = read(path)
        if declared is not None:
            return _applicable_requirement_names(declared)
    return set() if metadata_found else None


# ---------------------------------------------------------------------------
# What those declarations make importable
# ---------------------------------------------------------------------------


def _direct_requirements(dist_name: str) -> Set[str]:
    """Runtime requirements of an installed distribution (extras excluded)."""
    try:
        specifications = distribution(dist_name).requires or ()
    except PackageNotFoundError:
        return set()
    return _applicable_requirement_names(specifications)


def _dependency_closure(requirements: Iterable[str]) -> Set[str]:
    """*requirements* plus everything they pull in, transitively.

    A clean install receives the whole closure, so an import satisfied by a
    dependency's own dependency still works there. Only an import satisfied by
    something outside the closure is a packaging bug.
    """
    closure: Set[str] = set()
    pending = [canonicalize_name(name) for name in requirements]
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        closure.add(name)
        pending.extend(canonicalize_name(r) for r in _direct_requirements(name))
    return closure


def _roots_by_distribution() -> Dict[str, Set[str]]:
    """Canonical distribution name -> the top-level import names it provides."""
    roots: Dict[str, Set[str]] = {}
    for root, dist_names in packages_distributions().items():
        for dist_name in dist_names:
            roots.setdefault(canonicalize_name(dist_name), set()).add(root)
    return roots


def _top_level_txt_roots(dist_name: str) -> Set[str]:
    """Import names listed in a distribution's legacy ``top_level.txt``."""
    try:
        listing = distribution(dist_name).read_text("top_level.txt") or ""
    except PackageNotFoundError:
        return set()
    return {line.strip() for line in listing.splitlines() if line.strip()}


def _importable_roots(dist_names: Iterable[str]) -> Set[str]:
    """Top-level import names provided by *dist_names*.

    Three sources are unioned because none of them is complete on its own:
    ``packages_distributions`` is namespace-aware but misses some editable
    installs, ``top_level.txt`` is absent from modern wheels, and the
    normalized distribution name is all there is for something that is not
    installed here at all. Over-approximating is the safe direction -- a name
    that is wrongly allowed can only hide a bug, never invent one.
    """
    by_distribution = _roots_by_distribution()
    roots = set()
    for dist_name in dist_names:
        canonical = canonicalize_name(dist_name)
        roots.add(canonical.replace("-", "_"))
        roots |= by_distribution.get(canonical, set())
        roots |= _top_level_txt_roots(dist_name)
    return roots


# ---------------------------------------------------------------------------
# Importing the package, with and without the restriction
# ---------------------------------------------------------------------------

# Both probes run in a subprocess: blocking imports is process-wide, and the
# registry the package builds at import time is a mutable global that this very
# suite's doctests add to, so only a fresh interpreter reports what an install
# really offers. The probe reads its request as JSON from the command line, so
# this program text needs no interpolation and stays valid Python as written.
PROBE_PROGRAM = '''
import importlib
import json
import sys
from importlib.abc import MetaPathFinder

request = json.loads(sys.argv[1])
sys.path.insert(0, request["repo_dir"])
allowed_roots = request["allowed_roots"]

if allowed_roots is not None:
    allowed = frozenset(allowed_roots)

    class UndeclaredDistributionsAreUninstalled(MetaPathFinder):
        """Make every distribution outside the declared closure look uninstalled."""

        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".", 1)[0]
            if root in allowed:
                return None
            raise ModuleNotFoundError(
                "No module named " + repr(fullname) + " (not in install_requires)",
                name=fullname,
            )

    sys.meta_path.insert(0, UndeclaredDistributionsAreUninstalled())

package = importlib.import_module(request["package"])

json.dump(sorted(package.graph_kinds()), sys.stdout)
'''


@dataclass(frozen=True)
class ImportProbe:
    """What importing the package in a fresh interpreter gave."""

    imported: bool
    graph_kinds: FrozenSet[str]
    report: str


def _probe_import(
    *, repo_dir: Path, allowed_roots: Optional[Iterable[str]] = None
) -> ImportProbe:
    """Import the package in a fresh interpreter and report what it offers.

    With *allowed_roots* given, every top-level import outside that set is made
    to look uninstalled, which is what a clean install of the declared
    dependencies amounts to. Without it, the ambient environment is used as-is.
    """
    request = json.dumps(
        {
            "allowed_roots": None if allowed_roots is None else sorted(allowed_roots),
            "repo_dir": str(repo_dir),
            "package": PACKAGE_NAME,
        }
    )
    process = subprocess.run(
        [sys.executable, "-c", PROBE_PROGRAM, request],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        encoding=ENCODING,
    )
    if process.returncode != 0:
        return ImportProbe(False, frozenset(), process.stderr.strip())
    return ImportProbe(True, frozenset(json.loads(process.stdout)), "")


@pytest.fixture(scope="module")
def clean_install() -> ImportProbe:
    """The package imported with only its declared dependencies available."""
    requirements = _declared_requirements(REPO_DIR)
    if requirements is None:
        pytest.skip("no packaging metadata beside the package: not a source checkout")
    allowed = (
        _importable_roots(_dependency_closure(requirements))
        | set(sys.stdlib_module_names)
        | {PACKAGE_NAME}
    )
    return _probe_import(repo_dir=REPO_DIR, allowed_roots=allowed)


@pytest.fixture(scope="module")
def ambient_install() -> ImportProbe:
    """The package imported with everything this environment happens to have."""
    return _probe_import(repo_dir=REPO_DIR)


def test_import_needs_only_declared_dependencies(clean_install):
    """Importing the package must not reach outside its declared dependencies."""
    assert clean_install.imported, (
        "importing the package with only its declared dependencies failed; "
        "whatever it could not find belongs in install_requires:\n"
        f"{clean_install.report}"
    )


def test_declared_dependencies_provide_every_graph_kind(clean_install, ambient_install):
    """No registered graph kind may depend on an undeclared distribution."""
    if not (clean_install.imported and ambient_install.imported):
        pytest.skip("the package does not import; that failure is reported separately")
    missing = ambient_install.graph_kinds - clean_install.graph_kinds
    assert not missing, (
        "graph kinds available here but not after a clean install, because the "
        f"distribution they need is missing from install_requires: {sorted(missing)}"
    )
