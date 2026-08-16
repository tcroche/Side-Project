"""The pre-publication check has to fail on purpose before it is trusted.

Every check in check_repo.py is exercised against a tree built for it: a clean
tree must pass, and each planted mistake must be caught by its own check and by
no other. That is the difference between a script that runs and a script that
protects: the second one is the only kind worth putting in front of a first
public commit.
"""

from __future__ import annotations

import os

import pytest

import check_repo


def build_clean_repo(root) -> None:
    """The minimum shape check_repo expects to find in a healthy repository."""
    (root / ".gitignore").write_text(
        "\n".join([".env", "data/*.pkl", "data/*.csv", "data/llm_cache/", "/*.html"]),
        encoding="utf-8")
    (root / ".env.example").write_text(
        "# copy to .env\nANTHROPIC_API_KEY=sk-ant-...\n", encoding="utf-8")
    (root / "run_report.py").write_text("print('hello')\n", encoding="utf-8")
    (root / "README.md").write_text("# Project\n", encoding="utf-8")
    (root / "LICENSE").write_text("MIT License\n\n" + "Permission is hereby "
                                  "granted, free of charge, to any person "
                                  "obtaining a copy of this software.\n" * 3,
                                  encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / ".gitkeep").write_text("", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "writeup.pdf").write_bytes(b"%PDF-1.4\n")


def results(root) -> dict[str, check_repo.Check]:
    checks, _ = check_repo.run_all(str(root))
    return {c.name: c for c in checks}


@pytest.fixture
def clean(tmp_path):
    build_clean_repo(tmp_path)
    return tmp_path


def test_a_clean_repository_passes_every_check(clean):
    for name, check in results(clean).items():
        assert check.ok, f"{name} failed on a clean tree: {check.detail} {check.offenders}"


def test_the_placeholder_in_env_example_is_not_a_secret(clean):
    """`sk-ant-...` must not fire, or the check becomes noise and gets ignored."""
    assert results(clean)["secrets"].ok


#: Assembled at run time on purpose. A credential-shaped literal sitting in this
#: file would be a real finding for the very checker under test, and exempting
#: the file instead would put a hole in the scanner to make its test pass.
FAKE_ANTHROPIC = "sk-" + "ant-api03-" + "Zx9QpL2mNv7TrKa1BdEf"
FAKE_AWS = "AKIA" + "IOSFODNN7EXAMPLE"
FAKE_GITHUB = "ghp" + "_1234567890abcdefghijklmnopqrstuvwx"
FAKE_PEM = "-----BEGIN RSA PRIVATE" + " KEY-----"
WINDOWS_PATH = "C:" + "\\" + "Users" + "\\" + "theol" + "\\" + "Polymer"


def test_a_real_looking_key_is_caught_with_its_line(clean):
    (clean / "notebook.py").write_text(
        f"\n\nkey = '{FAKE_ANTHROPIC}'\n", encoding="utf-8")
    check = results(clean)["secrets"]
    assert not check.ok
    assert any("notebook.py:3" in o and "Anthropic" in o for o in check.offenders)


@pytest.mark.parametrize("payload", [FAKE_AWS, FAKE_GITHUB, FAKE_PEM])
def test_other_credential_shapes_are_caught_too(clean, payload):
    (clean / "leak.txt").write_text(payload, encoding="utf-8")
    assert not results(clean)["secrets"].ok


def test_a_secret_in_a_file_with_no_extension_is_caught(clean):
    """An extension allow-list would have skipped LICENSE, Dockerfile and any
    file simply called `key`."""
    (clean / "credentials").write_text(FAKE_ANTHROPIC, encoding="utf-8")
    assert not results(clean)["secrets"].ok


def test_binary_files_are_not_scanned_as_text(clean):
    (clean / "docs" / "diagram.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    assert results(clean)["secrets"].ok
    assert results(clean)["paths"].ok


def test_a_committed_env_file_is_caught(clean):
    (clean / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-real\n", encoding="utf-8")
    assert not results(clean)["env"].ok


def test_datasets_and_trial_exports_are_caught(clean):
    (clean / "data" / "trials_m2_with_rut.csv").write_text("a,b\n", encoding="utf-8")
    (clean / "prices.pkl").write_bytes(b"\x80\x04")
    check = results(clean)["data"]
    assert not check.ok
    assert len(check.offenders) == 2


def test_the_model_cache_is_caught(clean):
    cache = clean / "data" / "llm_cache"
    cache.mkdir()
    (cache / "abc.json").write_text("{}", encoding="utf-8")
    assert not results(clean)["data"].ok


def test_a_leftover_fixture_is_caught_anywhere_in_the_tree(clean):
    """Rule 9. The fixture check looks at the whole tree, not only at what would
    be published: an ignored fixture is still a file a command can pick up."""
    (clean / "data" / "dryrun_trials_m2_meta.csv").write_text("x\n", encoding="utf-8")
    check = results(clean)["fixture"]
    assert not check.ok
    assert "data/dryrun_trials_m2_meta.csv" in check.offenders


def test_a_generated_report_at_the_root_is_caught_but_docs_is_allowed(clean):
    (clean / "docs" / "example_report.html").write_text("<p>x</p>", encoding="utf-8")
    assert results(clean)["generated"].ok, "publishing an example under docs/ is fine"
    (clean / "rapport_m2.html").write_text("<p>x</p>", encoding="utf-8")
    assert not results(clean)["generated"].ok


def test_a_gitignore_missing_a_required_rule_is_caught(clean):
    (clean / ".gitignore").write_text(".env\ndata/*.pkl\n", encoding="utf-8")
    check = results(clean)["ignore-rules"]
    assert not check.ok
    assert "data/*.csv" in check.offenders and "data/llm_cache/" in check.offenders


POSIX_PATH = "/home/" + "theol/Polymer/data"


@pytest.mark.parametrize("written", [
    WINDOWS_PATH,                        # escaped form, two separators
    WINDOWS_PATH.replace("\\\\", "\\"),    # raw form, one separator
    POSIX_PATH,
])
def test_absolute_machine_paths_are_caught(clean, written):
    (clean / "helper.py").write_text(f'PATH = "{written}"\n', encoding="utf-8")
    check = results(clean)["paths"]
    assert not check.ok
    assert any("helper.py" in o for o in check.offenders)


def test_the_checker_does_not_report_its_own_example_paths(clean):
    """check_repo.py contains the shapes it looks for, by necessity."""
    source = os.path.join(check_repo.ROOT, "check_repo.py")
    with open(source, "r", encoding="utf-8") as handle:
        (clean / "check_repo.py").write_text(handle.read(), encoding="utf-8")
    assert results(clean)["paths"].ok


def test_a_missing_writeup_is_reported(clean):
    os.remove(clean / "docs" / "writeup.pdf")
    check = results(clean)["writeup"]
    assert not check.ok
    assert "build_writeup.py" in check.detail


def test_a_missing_licence_is_reported(clean):
    os.remove(clean / "LICENSE")
    check = results(clean)["license"]
    assert not check.ok
    assert "all rights reserved" in check.detail


def test_a_truncated_licence_is_reported(clean):
    (clean / "LICENSE").write_text("MIT\n", encoding="utf-8")
    assert not results(clean)["license"].ok


def test_an_unexpectedly_large_file_is_reported(clean):
    (clean / "big.md").write_text("x" * (check_repo.LARGE_FILE_BYTES + 1), encoding="utf-8")
    check = results(clean)["size"]
    assert not check.ok
    assert any("big.md" in o for o in check.offenders)


def test_the_real_repository_passes(clean):
    """The check is only worth having if the repository it guards is clean now."""
    checks, _ = check_repo.run_all(check_repo.ROOT)
    failed = [f"{c.name}: {c.detail} {c.offenders}" for c in checks if not c.ok]
    assert not failed, "\n".join(failed)