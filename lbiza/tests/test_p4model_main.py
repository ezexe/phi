"""Tests for python -m phi_model's choice of an environment: the store's own first, a worktree's main checkout's,
and the older places after it."""
import os

from phi_model import __main__ as m

REL = os.path.join("Scripts", "python.exe") if os.name == "nt" else os.path.join("bin", "python")


def interpreter(venv):
    """An environment at `venv` with its interpreter, an empty file, which is all the fallback looks at."""
    py = venv / REL
    py.parent.mkdir(parents=True)
    py.write_text("")
    return str(py)


def test_the_argument_is_read_in_both_forms():
    assert m.project_arg(["check", "--project", "a/b"]) == "a/b"
    assert m.project_arg(["check", "--project=a/b", "--json"]) == "a/b"
    assert m.project_arg(["check"]) == ""
    assert m.project_arg(["check", "--project"]) == ""


def test_the_stores_environment_comes_first(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    py = interpreter(proj / ".claude" / "p4" / ".venv")
    assert m.venv_python(["check", "--project", str(proj)]) == py
    assert m.venv_python(["check", f"--project={proj / '.claude' / 'p4'}"]) == py


def test_a_worktree_takes_the_main_checkouts(tmp_path):
    main, wt = tmp_path / "main", tmp_path / "wt"
    admin = main / ".git" / "worktrees" / "wt"
    admin.mkdir(parents=True)
    (admin / "commondir").write_text("../..\n", encoding="utf-8")
    (wt / "sub").mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {admin}\n", encoding="utf-8")
    assert m.main_checkout(str(wt / "sub")) == str(main)
    assert m.store_venv(str(wt / ".claude" / "p4")) == str(main / ".claude" / "p4" / ".venv")
    py = interpreter(main / ".claude" / "p4" / ".venv")
    assert m.venv_python(["check", "--project", str(wt)]) == py


def test_a_store_without_an_environment_leaves_the_older_places(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    assert m.venv_python(["check", "--project", str(proj)]) == m.venv_python(["check"])
