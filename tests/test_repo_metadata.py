"""Tests for the repo metadata/docs added in this PR: the GitHub issue
templates, CITATION.cff, and the new/changed cross-references in README.md.

These are plain-text/config files rather than Python modules, so the tests
parse them directly. PyYAML is *not* in requirements-dev.txt or the CI
workflow's install step, so the primary assertions use a small hand-rolled
parser for the flat `key: value` (+ simple list) subset of YAML these files
use. Where PyYAML happens to be installed, an extra test cross-checks the
same files with a real YAML parser.
"""
import datetime
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ISSUE_TEMPLATE_DIR = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"
BUG_REPORT = ISSUE_TEMPLATE_DIR / "bug_report.md"
FEATURE_REQUEST = ISSUE_TEMPLATE_DIR / "feature_request.md"
CITATION = REPO_ROOT / "CITATION.cff"
README = REPO_ROOT / "README.md"


def _split_frontmatter(text):
    """Split a file starting with `---\\n...\\n---\\n` into (frontmatter_lines, body)."""
    lines = text.splitlines()
    assert lines and lines[0] == "---", "file must start with a YAML frontmatter delimiter"
    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line == "---":
            end = i
            break
    assert end is not None, "frontmatter is never closed with a second '---' line"
    return lines[1:end], "\n".join(lines[end + 1:])


def _parse_flat_yaml(lines):
    """Minimal parser for the flat `key: value` (+ simple `- item` list) subset
    of YAML used by these files. Deliberately not a general YAML parser."""
    data = {}
    current_list_key = None
    for raw in lines:
        if not raw.strip():
            continue
        stripped = raw.strip()
        if stripped.startswith("- "):
            assert current_list_key is not None, f"list item outside of a list key: {raw!r}"
            item = stripped[2:].strip().strip('"')
            data.setdefault(current_list_key, []).append(item)
            continue
        key, sep, value = raw.partition(":")
        assert sep == ":", f"expected a 'key: value' line, got {raw!r}"
        key = key.strip()
        value = value.strip()
        if value == "":
            current_list_key = key
            data[key] = []
        else:
            current_list_key = None
            data[key] = value.strip('"')
    return data


def test_split_frontmatter_requires_opening_delimiter():
    with pytest.raises(AssertionError):
        _split_frontmatter("name: x\n---\n")


def test_split_frontmatter_requires_closing_delimiter():
    with pytest.raises(AssertionError):
        _split_frontmatter("---\nname: x\n")


# ---------------------------------------------------------------------------
# .github/ISSUE_TEMPLATE/bug_report.md
# ---------------------------------------------------------------------------

def test_bug_report_exists():
    assert BUG_REPORT.is_file()


def test_bug_report_frontmatter_fields():
    front, _ = _split_frontmatter(BUG_REPORT.read_text())
    fm = _parse_flat_yaml(front)
    assert fm["name"] == "Bug report"
    assert fm["about"] == "Something crashed, hung, or produced wrong results during training"
    assert fm["title"] == "[bug] "
    assert fm["labels"] == "bug"


def test_bug_report_body_has_required_sections():
    _, body = _split_frontmatter(BUG_REPORT.read_text())
    for heading in (
        "## What happened",
        "## Command you ran",
        "## Environment",
        "## Logs",
        "## Scheduler state (if relevant)",
    ):
        assert heading in body, f"missing section: {heading}"


def test_bug_report_has_balanced_code_fences():
    _, body = _split_frontmatter(BUG_REPORT.read_text())
    assert "```bash" in body
    assert "```text" in body
    assert body.count("```") % 2 == 0, "unbalanced code fence in bug_report.md"


def test_bug_report_environment_checklist_fields():
    _, body = _split_frontmatter(BUG_REPORT.read_text())
    for field in ("Model ID:", "GPU (VRAM):", "OS:", "`torch` / `transformers` versions:"):
        assert field in body


def test_bug_report_mentions_scheduler_checkpoint_state():
    _, body = _split_frontmatter(BUG_REPORT.read_text())
    assert "checkpoint_full.pt" in body
    assert "scheduler_state" in body


# ---------------------------------------------------------------------------
# .github/ISSUE_TEMPLATE/feature_request.md
# ---------------------------------------------------------------------------

def test_feature_request_exists():
    assert FEATURE_REQUEST.is_file()


def test_feature_request_frontmatter_fields():
    front, _ = _split_frontmatter(FEATURE_REQUEST.read_text())
    fm = _parse_flat_yaml(front)
    assert fm["name"] == "Feature request"
    assert fm["about"] == "Suggest an improvement or new capability"
    assert fm["title"] == "[feature] "
    assert fm["labels"] == "enhancement"


def test_feature_request_body_has_required_sections():
    _, body = _split_frontmatter(FEATURE_REQUEST.read_text())
    for heading in ("## What problem does this solve?", "## Proposed change"):
        assert heading in body


def test_feature_request_references_contributing_ground_rules():
    _, body = _split_frontmatter(FEATURE_REQUEST.read_text())
    assert "CONTRIBUTING.md" in body


def test_issue_templates_use_distinct_title_prefix_and_labels():
    # the two templates must not collide on their auto-filled title tag/label
    bug_fm = _parse_flat_yaml(_split_frontmatter(BUG_REPORT.read_text())[0])
    feat_fm = _parse_flat_yaml(_split_frontmatter(FEATURE_REQUEST.read_text())[0])
    assert bug_fm["title"] != feat_fm["title"]
    assert bug_fm["labels"] != feat_fm["labels"]


# ---------------------------------------------------------------------------
# CITATION.cff
# ---------------------------------------------------------------------------

REQUIRED_CFF_KEYS = (
    "cff-version", "message", "title", "version", "date-released",
    "authors", "url", "repository-code", "license", "keywords",
)


def _load_citation():
    return _parse_flat_yaml(CITATION.read_text().splitlines())


def test_citation_exists():
    assert CITATION.is_file()


def test_citation_has_all_required_keys():
    cff = _load_citation()
    for key in REQUIRED_CFF_KEYS:
        assert key in cff, f"CITATION.cff missing required key: {key}"


def test_citation_cff_version_is_well_formed():
    cff = _load_citation()
    assert re.fullmatch(r"\d+\.\d+\.\d+", cff["cff-version"])


def test_citation_version_matches_pyproject_version():
    cff = _load_citation()
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert m, "could not find [project].version in pyproject.toml"
    assert cff["version"] == m.group(1)


def test_citation_date_released_is_a_valid_iso_date():
    cff = _load_citation()
    parsed = datetime.date.fromisoformat(cff["date-released"])
    assert parsed <= datetime.date.today(), "date-released should not be in the future"


def test_citation_urls_point_at_the_repository():
    cff = _load_citation()
    for key in ("url", "repository-code"):
        assert cff[key].startswith("https://github.com/")
    assert cff["url"] == cff["repository-code"]


def test_citation_license_is_mit_and_matches_license_file():
    cff = _load_citation()
    assert cff["license"] == "MIT"
    assert "MIT License" in (REPO_ROOT / "LICENSE").read_text()


def test_citation_authors_nonempty():
    cff = _load_citation()
    assert len(cff["authors"]) >= 1
    assert any("name:" in a for a in cff["authors"])


def test_citation_keywords_nonempty_and_unique():
    cff = _load_citation()
    keywords = cff["keywords"]
    assert len(keywords) >= 1
    assert len(keywords) == len(set(keywords)), "duplicate keyword entries"


def test_citation_title_mentions_project_name():
    cff = _load_citation()
    assert "event-aware-SAE-trainer" in cff["title"]


def test_citation_message_is_a_citation_instruction():
    cff = _load_citation()
    assert "cite" in cff["message"].lower()


def test_citation_is_consistent_with_real_yaml_parser_when_available():
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(CITATION.read_text())
    assert data["cff-version"] == "1.2.0"
    assert data["version"] == "0.8.0"
    assert isinstance(data["authors"], list) and len(data["authors"]) == 1
    assert data["authors"][0]["name"] == "Rick (JuiceB0xC0de)"
    assert isinstance(data["keywords"], list) and len(data["keywords"]) == 5


# ---------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------

_LOCAL_LINK_RE = re.compile(r"\]\(([^)]+)\)")


def _readme_text():
    return README.read_text()


def _local_readme_links():
    links = []
    for target in _LOCAL_LINK_RE.findall(_readme_text()):
        if target.startswith("http://") or target.startswith("https://"):
            continue
        links.append(target)
    return links


def test_readme_exists():
    assert README.is_file()


def test_readme_has_local_links():
    assert _local_readme_links(), "expected README.md to contain at least one local markdown link"


def test_readme_local_links_resolve_to_real_files():
    for target in _local_readme_links():
        path = target.split("#", 1)[0]  # strip any in-page anchor
        resolved = (REPO_ROOT / path).resolve()
        assert resolved.is_file(), f"README.md links to a missing file: {target}"


def test_readme_references_new_docs_added_by_this_pr():
    text = _readme_text()
    for doc in ("EFFICIENCY.md", "CHANGELOG.md", "CONTRIBUTING.md", "CITATION.cff"):
        assert doc in text


def test_readme_citation_section_present():
    text = _readme_text()
    assert "## Citation" in text
    assert "CITATION.cff" in text.split("## Citation", 1)[1]


def test_readme_tests_badge_points_to_an_existing_workflow_file():
    text = _readme_text()
    m = re.search(r"\[!\[tests\]\(([^)]+)\)\]\(([^)]+)\)", text)
    assert m, "expected a tests badge with an image target and a link target"
    badge_url, link_url = m.groups()
    prefix = "https://github.com/JuiceB0xC0de/event-aware-SAE-trainer/actions/workflows/"
    assert badge_url.startswith(prefix)
    assert link_url.startswith(prefix)
    workflow_file = badge_url[len(prefix):].split("/", 1)[0]
    assert workflow_file == link_url[len(prefix):]
    assert (REPO_ROOT / ".github" / "workflows" / workflow_file).is_file()


def test_readme_use_trained_sae_example_referenced_and_present():
    text = _readme_text()
    assert "examples/use_trained_sae.py" in text
    assert (REPO_ROOT / "examples" / "use_trained_sae.py").is_file()


def test_readme_hardware_table_lists_three_presets():
    text = _readme_text()
    for row_marker in ("H100 / A100 pod", "RTX 4090", "RTX 3080"):
        assert row_marker in text


def test_readme_resume_section_mentions_pool_regeneration_avoidance():
    text = _readme_text()
    assert "regenerating pools from layer 0" in text


def test_readme_mentions_ci_runs_on_push_and_pr():
    text = _readme_text()
    assert "CI runs the suite on every push and PR" in text