"""The docs site builds: every doc, notebook, component and artifact gets a page, the index lists them all, and a
doc's relative links land on the site or on GitHub, never on a path the site doesn't carry."""

import os
import re

import pytest

pytest.importorskip("markdown")
pytest.importorskip("nbconvert")


def test_site_builds(tmp_path):
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("site_build", os.path.join(root, "scripts", "site", "build.py"))
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    out = tmp_path / "site"
    listed, missing = build.build(str(out))
    assert not missing
    kinds = {k for _, _, k in listed}
    assert kinds == {"doc", "notebook", "interactive", "artifact"}
    index = (out / "index.html").read_text(encoding="utf-8")
    for site, _, _ in listed:
        assert (out / site).exists() and f'href="{site}"' in index
    html = (out / "docs" / "register.html").read_text(encoding="utf-8")
    for href in re.findall(r'href="([^"#:]+)"', html):
        assert (out / "docs" / href).exists(), href
