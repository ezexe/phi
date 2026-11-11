"""python -m phi_model. Started by an interpreter without numpy (the system python, where the package lives in a
checkout's .venv), it runs itself again in that .venv instead of checking and training with half a package."""
import importlib.util
import os
import subprocess
import sys


def venv_python():
    """The first .venv python that exists: this checkout's, the one beside it, or a sibling p4 checkout's
    (scripts/make_venv.py builds that one)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    up = os.path.dirname(repo)
    rel = os.path.join("Scripts", "python.exe") if os.name == "nt" else os.path.join("bin", "python")
    for root in (repo, up, os.path.join(up, "p4")):
        py = os.path.join(root, ".venv", rel)
        if os.path.isfile(py):
            return py
    return None


def rerun_in_venv():
    """The .venv's exit code when this interpreter lacks numpy and a .venv exists; None to carry on here."""
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
