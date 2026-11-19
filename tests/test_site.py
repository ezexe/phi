"""The docs site builds: every doc, notebook and hand-written page gets a page, the index lists
them all, and a doc's relative links land on the site or on GitHub, never on a path the site doesn't carry."""

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
    assert kinds == {"doc", "page"}
    index = (out / "index.html").read_text(encoding="utf-8")
    for site, _, _ in listed:
        assert (out / site).exists() and f'href="{site}"' in index
    for site, _, _ in listed:
        folder = (out / site).parent
        if folder != out:
            assert (folder / "index.html").exists(), folder
    phi = (out / "docs" / "phi" / "index.html").read_text(encoding="utf-8")
    assert 'href="phi.html"' in phi
    assert ('docs/phi/codes.html', 'Fibonacci codes · phi docs', 'page') in listed and 'href="codes.html"' in phi
    assert 'href="phi/index.html"' in (out / "docs" / "index.html").read_text(encoding="utf-8")
    html = (out / "docs" / "register.html").read_text(encoding="utf-8")
    for href in re.findall(r'href="([^"#:]+)"', html):
        assert (out / "docs" / href).exists(), href
