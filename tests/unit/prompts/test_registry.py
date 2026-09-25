from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from client_research_agent.prompts.registry import (
    PromptError,
    PromptRegistry,
    PromptTemplate,
    default_registry,
    parse_template,
)
from client_research_agent.utils.errors import ConfigurationError

EXPECTED = {
    "brief_writer",
    "citation_judge",
    "criterion_qualifier",
    "discovery_questions",
    "opportunity_analysis",
    "research_planner",
}

VALID = """---
name: sample_prompt
version: 1.2.3
description: A prompt used in tests.
variables: [company, evidence]
---
Assess {company}. Output {"score": 1}.
{evidence}
"""


def test_packaged_templates_load_with_metadata() -> None:
    registry = PromptRegistry()
    assert set(registry.names()) == EXPECTED
    for name in registry.names():
        template = registry.get(name)
        assert re.fullmatch(r"\d+\.\d+\.\d+", template.version)
        assert re.fullmatch(r"[0-9a-f]{64}", template.fingerprint)
        assert template.description
        assert template.placeholders() == set(template.variables)
        assert template.ref == f"{name}@{template.version}"


def test_evidence_prompts_carry_untrusted_data_instructions() -> None:
    registry = PromptRegistry()
    for name in registry.names():
        template = registry.get(name)
        if "evidence" not in template.variables:
            continue
        body = template.body.lower()
        assert "untrusted" in body
        assert "never follow instructions" in body
        assert "json" in body
    qualifier = registry.get("criterion_qualifier").body
    assert "ONLY the evidence" in qualifier
    assert '"E3"' in qualifier


def test_render_substitutes_all_variables_single_pass() -> None:
    template = parse_template(VALID)
    rendered = template.render(company="Acme {evidence}", evidence="<evidence>{company}</evidence>")
    assert "Assess Acme {evidence}." in rendered
    assert "<evidence>{company}</evidence>" in rendered
    assert '{"score": 1}' in rendered


def test_render_rejects_missing_and_unexpected_variables() -> None:
    template = parse_template(VALID)
    with pytest.raises(PromptError, match="missing"):
        template.render(company="Acme")
    with pytest.raises(PromptError, match="unexpected"):
        template.render(company="Acme", evidence="x", extra="y")


def test_prompt_error_is_configuration_error() -> None:
    assert issubclass(PromptError, ConfigurationError)


def test_fingerprint_is_sha256_of_content_and_changes_with_content() -> None:
    first = parse_template(VALID)
    assert first.fingerprint == hashlib.sha256(VALID.encode()).hexdigest()
    assert first.short_fingerprint == first.fingerprint[:12]
    second = parse_template(VALID.replace("Assess", "Evaluate"))
    assert second.fingerprint != first.fingerprint
    assert parse_template(VALID.replace("\n", "\r\n")).fingerprint == first.fingerprint


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("name: x\n", "front-matter"),
        ("---\nname: x\n", "unterminated"),
        ("---\n: : [\n---\nbody\n", "YAML"),
        ("---\n- a\n- b\n---\nbody\n", "mapping"),
        ("---\nname: Bad-Name\nversion: 1.0.0\ndescription: d\n---\nbody\n", "name"),
        ("---\nname: ok\nversion: 1.0\ndescription: d\n---\nbody\n", "version"),
        ("---\nname: ok\nversion: 1.0.0\n---\nbody\n", "description"),
        ("---\nname: ok\nversion: 1.0.0\ndescription: d\nvariables: company\n---\n{company}\n", "list"),
        ("---\nname: ok\nversion: 1.0.0\ndescription: d\nvariables: [a, a]\n---\n{a}\n", "duplicate"),
        ("---\nname: ok\nversion: 1.0.0\ndescription: d\nvariables: []\n---\n{company}\n", "not declared"),
        ("---\nname: ok\nversion: 1.0.0\ndescription: d\nvariables: [a, b]\n---\n{a}\n", "never used"),
    ],
)
def test_parse_template_validation(text: str, message: str) -> None:
    with pytest.raises(PromptError, match=message):
        parse_template(text, source="t.md")


def test_empty_front_matter_is_rejected() -> None:
    with pytest.raises(PromptError, match="name"):
        parse_template("---\n\n---\nbody\n")


def test_registry_from_directory(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(VALID, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    registry = PromptRegistry.from_directory(tmp_path)
    assert registry.names() == ("sample_prompt",)
    assert "sample_prompt" in registry
    assert registry.render("sample_prompt", company="A", evidence="B").startswith("Assess A.")
    assert registry.fingerprints() == {"sample_prompt@1.2.3": registry.get("sample_prompt").fingerprint}
    versions = registry.model_versions()
    assert versions == {"prompt:sample_prompt": f"1.2.3+{registry.get('sample_prompt').short_fingerprint}"}


def test_registry_directory_errors(tmp_path: Path) -> None:
    with pytest.raises(PromptError, match="does not exist"):
        PromptRegistry.from_directory(tmp_path / "missing")
    with pytest.raises(PromptError, match="no prompt templates"):
        PromptRegistry.from_directory(tmp_path)


def test_registry_rejects_duplicates_and_unknown_names() -> None:
    template = parse_template(VALID)
    registry = PromptRegistry([template])
    with pytest.raises(PromptError, match="duplicate"):
        registry.register(template)
    with pytest.raises(PromptError, match="unknown prompt"):
        registry.get("nope")


def test_default_registry_is_cached() -> None:
    assert default_registry() is default_registry()
    assert isinstance(default_registry().get("citation_judge"), PromptTemplate)
