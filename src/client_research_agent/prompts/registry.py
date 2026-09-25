"""Versioned prompt templates with content fingerprints.

Templates live in ``client_research_agent/prompts/templates/*.md`` and ship in
the wheel. Each file starts with YAML front-matter::

    ---
    name: criterion_qualifier
    version: 1.2.0
    description: Scores one qualification criterion from evidence.
    variables: [company, criterion_title, evidence]
    ---
    <template body with {company} style placeholders>

Rendering is a *safe* subset of ``str.format``: only bare ``{identifier}``
placeholders declared in ``variables`` are substituted, in a single pass, so
values containing braces (JSON, untrusted evidence) are inserted literally and
are never re-expanded; attribute/index access (``{x.__class__}``) is not
supported at all. Every declared variable must be supplied and every
placeholder in the body must be declared, which is checked at load time.

``PromptTemplate.fingerprint`` is the SHA-256 of the raw file content and is
logged to MLflow with each run so any brief can be traced to the exact prompt
text that produced it.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import yaml

from client_research_agent.utils.errors import ConfigurationError

_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
TEMPLATE_PACKAGE = "client_research_agent.prompts"
TEMPLATE_DIR = "templates"


class PromptError(ConfigurationError):
    """A prompt template is malformed, missing, or rendered with the wrong variables."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    version: str
    description: str
    variables: tuple[str, ...]
    body: str
    fingerprint: str = field(repr=False)

    @property
    def ref(self) -> str:
        """``name@version`` identifier used in logs and model version maps."""
        return f"{self.name}@{self.version}"

    @property
    def short_fingerprint(self) -> str:
        return self.fingerprint[:12]

    def placeholders(self) -> set[str]:
        return set(_PLACEHOLDER.findall(self.body))

    def render(self, **values: Any) -> str:
        missing = [name for name in self.variables if name not in values]
        if missing:
            raise PromptError(f"prompt {self.ref}: missing variables {missing}")
        unexpected = sorted(set(values) - set(self.variables))
        if unexpected:
            raise PromptError(f"prompt {self.ref}: unexpected variables {unexpected}")
        rendered = {name: str(values[name]) for name in self.variables}
        return _PLACEHOLDER.sub(lambda match: rendered.get(match.group(1), match.group(0)), self.body)


def parse_template(text: str, *, source: str = "<string>") -> PromptTemplate:
    """Parse a template file's content (front-matter + body) and validate it."""
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        raise PromptError(f"{source}: template must start with '---' front-matter")
    try:
        _, header, body = normalized.split("---\n", 2)
    except ValueError as exc:
        raise PromptError(f"{source}: unterminated front-matter") from exc
    try:
        meta = yaml.safe_load(header) or {}
    except yaml.YAMLError as exc:
        raise PromptError(f"{source}: invalid front-matter YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise PromptError(f"{source}: front-matter must be a mapping")
    name = str(meta.get("name", ""))
    version = str(meta.get("version", ""))
    description = str(meta.get("description", "")).strip()
    raw_variables = meta.get("variables") or []
    if not _NAME.match(name):
        raise PromptError(f"{source}: invalid or missing name {name!r}")
    if not _VERSION.match(version):
        raise PromptError(f"{source}: version must be semantic (x.y.z), got {version!r}")
    if not description:
        raise PromptError(f"{source}: description is required")
    if not isinstance(raw_variables, list) or not all(isinstance(v, str) for v in raw_variables):
        raise PromptError(f"{source}: variables must be a list of strings")
    variables = tuple(raw_variables)
    if len(set(variables)) != len(variables):
        raise PromptError(f"{source}: duplicate variables")
    body = body.strip() + "\n"
    used = set(_PLACEHOLDER.findall(body))
    undeclared = sorted(used - set(variables))
    unused = sorted(set(variables) - used)
    if undeclared:
        raise PromptError(f"{source}: placeholders not declared in variables: {undeclared}")
    if unused:
        raise PromptError(f"{source}: declared variables never used: {unused}")
    fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return PromptTemplate(
        name=name,
        version=version,
        description=description,
        variables=variables,
        body=body,
        fingerprint=fingerprint,
    )


class PromptRegistry:
    """Loads and serves prompt templates; safe to share across threads."""

    def __init__(self, templates: Iterable[PromptTemplate] | None = None) -> None:
        self._lock = threading.Lock()
        self._templates: dict[str, PromptTemplate] = {}
        for template in templates if templates is not None else _load_packaged():
            self.register(template)

    @classmethod
    def from_directory(cls, directory: Path | str) -> PromptRegistry:
        path = Path(directory)
        if not path.is_dir():
            raise PromptError(f"prompt directory {path} does not exist")
        return cls(_load_from(path))

    def register(self, template: PromptTemplate) -> None:
        with self._lock:
            if template.name in self._templates:
                raise PromptError(f"duplicate prompt name {template.name!r}")
            self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        with self._lock:
            template = self._templates.get(name)
        if template is None:
            raise PromptError(f"unknown prompt {name!r}; available: {sorted(self._templates)}")
        return template

    def render(self, name: str, **values: Any) -> str:
        return self.get(name).render(**values)

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._templates))

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._templates

    def fingerprints(self) -> Mapping[str, str]:
        """``{name@version: sha256}`` for every template, for MLflow lineage."""
        with self._lock:
            return {t.ref: t.fingerprint for t in self._templates.values()}

    def model_versions(self, names: Iterable[str] | None = None) -> dict[str, str]:
        """Compact ``{"prompt:<name>": "<version>+<sha12>"}`` entries for ``ClientBrief.model_versions``."""
        selected = self.names() if names is None else tuple(names)
        return {
            f"prompt:{name}": f"{self.get(name).version}+{self.get(name).short_fingerprint}"
            for name in selected
        }


def _load_from(directory: Traversable | Path) -> list[PromptTemplate]:
    templates = [
        parse_template(entry.read_text(encoding="utf-8"), source=entry.name)
        for entry in sorted(directory.iterdir(), key=lambda item: item.name)
        if entry.is_file() and entry.name.endswith(".md")
    ]
    if not templates:
        raise PromptError(f"no prompt templates found in {directory}")
    return templates


def _load_packaged() -> list[PromptTemplate]:
    return _load_from(resources.files(TEMPLATE_PACKAGE) / TEMPLATE_DIR)


_default_registry: PromptRegistry | None = None
_default_lock = threading.Lock()


def default_registry() -> PromptRegistry:
    """Process-wide registry of the packaged templates (loaded once)."""
    global _default_registry  # noqa: PLW0603 - lazily initialised singleton
    with _default_lock:
        if _default_registry is None:
            _default_registry = PromptRegistry()
        return _default_registry
