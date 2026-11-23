"""python -m phi_model. Started by an interpreter without numpy, such as the system python where the package lives
in an environment, it runs itself again in that environment instead of checking and training with half a package.
The first environment that exists is taken: the store's own, `.claude/p4/.venv` in the main checkout of the
repository --project names, which p4-phi's `p4.py venv` builds, and then this checkout's .venv, the one beside it,
or a sibling p4 checkout's."""
import importlib.util
import os
import subprocess
import sys


def project_arg(argv):
    """The value of --project among the arguments, as `--project X` or `--project=X`; '' when there is none."""
    for i, a in enumerate(argv):
        if a == "--project" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--project="):
            return a.split("=", 1)[1]
    return ""


def main_checkout(project):
    """The main checkout of the git repository holding `project`, read from the `.git` files without running git,
    as p4-phi's p4conf.main_root reads them: a worktree's `.git` file names its gitdir, whose `commondir` leads to
    the main `.git`. The folder itself outside a repository."""
    d = start = os.path.abspath(project)
    while True:
        dot = os.path.join(d, ".git")
        if os.path.isdir(dot):
            return d
        if os.path.isfile(dot):
            try:
                with open(dot, encoding="utf-8", errors="replace") as f:
                    gitdir = f.read().strip().split("gitdir:", 1)[1].strip()
                gitdir = os.path.normpath(os.path.join(d, gitdir))
                with open(os.path.join(gitdir, "commondir"), encoding="utf-8", errors="replace") as f:
                    return os.path.dirname(os.path.normpath(os.path.join(gitdir, f.read().strip())))
            except (OSError, IndexError):
                return d                                # a submodule, or a pointer that does not read: its own root
        up = os.path.dirname(d)
        if up == d:
            return start
        d = up


def store_venv(project):
    """The environment of the store of the repository `project` names, by itself or by its `.claude/p4`:
    `.claude/p4/.venv` in the repository's main checkout, so a worktree gives the main checkout's. A folder inside
    a store stands for the store's repository too, since the store is a git repository of its own."""
    base = d = os.path.abspath(project)
    while True:                                 # the nearest .claude/p4 at or above the folder: its repository
        up = os.path.dirname(d)
        if up == d:
            break
        if os.path.basename(d) == "p4" and os.path.basename(up) == ".claude":
            base = os.path.dirname(up)
            break
        d = up
    return os.path.join(main_checkout(base), ".claude", "p4", ".venv")


def venv_python(argv=None):
    """The first environment's python that exists: the store's own, of the repository --project names, then this
    checkout's .venv, the one beside it, or a sibling p4 checkout's."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    up = os.path.dirname(repo)
    rel = os.path.join("Scripts", "python.exe") if os.name == "nt" else os.path.join("bin", "python")
    venvs = [os.path.join(root, ".venv") for root in (repo, up, os.path.join(up, "p4"))]
    project = project_arg(sys.argv[1:] if argv is None else argv)
    if project:
        venvs.insert(0, store_venv(project))
    for venv in venvs:
        py = os.path.join(venv, rel)
        if os.path.isfile(py):
            return py
    return None


def rerun_in_venv():
    """The environment's exit code when this interpreter lacks numpy and an environment exists; None to carry on
    here."""
    if importlib.util.find_spec("numpy") is not None or os.environ.get("PHI_MODEL_REEXEC"):
        return None
    py = venv_python()
    if py is None or os.path.normcase(os.path.abspath(py)) == os.path.normcase(os.path.abspath(sys.executable)):
        return None
    print(f"phi_model: no numpy in {sys.executable}; running in {py}", file=sys.stderr, flush=True)
    return subprocess.call([py, "-m", "phi_model"] + sys.argv[1:], env=dict(os.environ, PHI_MODEL_REEXEC="1"))


if __name__ == "__main__":
    code = rerun_in_venv()
    if code is None:
        from phi_model import main
        code = main()
    sys.exit(code)
