"""Every link in the pages and the markdown files that names a file of this repository names one it holds.

The pages are the .html files under docs/, which .github/workflows/pages.yml publishes as they are at
https://ezexe.github.io/phi/, and the markdown files are every .md file git tracks, the README among them.
A link is an href or src attribute of a page, and in a markdown file the link destination of an inline link, an
autolink, a link reference definition, or an href or src attribute of raw HTML. A link names a file of this
repository in three ways, and each is resolved from the files alone, with no request sent:

    a relative URL                                 the file beside the one that holds the link; from a page it stays
                                                   under docs/, since the site holds nothing else
    https://ezexe.github.io/phi/<path>             docs/<path>
    https://github.com/ezexe/phi/blob/main/<path>  <path>, and the same under tree/main/, raw/main/ and blame/main/,
                                                   and at https://raw.githubusercontent.com/ezexe/phi/main/<path>

The repository holds what git tracks, so a file that only this machine has satisfies no link. A link to a folder
opens the folder's index.html on the site and the folder itself on GitHub, and a link that ends in a slash names a
folder. Names are compared by case, as the site and GitHub compare them. A link's fragment is not checked, and every
other link leaves the repository and is not followed.
"""
import functools
import pathlib
import posixpath
import re
import subprocess
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE_HOST, SITE_PATH = "ezexe.github.io", "/phi"
REPO_HOST, REPO_PATH = "github.com", "/ezexe/phi"
RAW_HOST = "raw.githubusercontent.com"
INLINE_LINK = re.compile(r"\]\(\s*(<[^>\n]*>|[^\s)]+)")
AUTOLINK = re.compile(r"<([A-Za-z][A-Za-z0-9+.-]*:[^\s<>]*)>")
REFERENCE = re.compile(r"^ {0,3}\[[^\]]+\]:\s*(<[^>\n]*>|\S+)")
ATTRIBUTE = re.compile(r"""\b(?:href|src)\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
CODE_SPAN = re.compile(r"(`+).*?\1")
FENCE = re.compile(r"^ {0,3}(```|~~~)")


class Attributes(HTMLParser):
    """The href and src attributes of a page, each with the line of its tag."""

    def __init__(self):
        super().__init__()
        self.found = []

    def handle_starttag(self, tag, attrs):
        line = self.getpos()[0]
        self.found += [(line, value) for name, value in attrs if name in ("href", "src") and value]


def page_links(path):
    """[(line, link)] of a page. A script's text is not markup, so a link a script writes is not read."""
    parser = Attributes()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser.found


def markdown_links(path):
    """[(line, link)] of a markdown file, outside its fenced code blocks and code spans."""
    found, fenced = [], False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        line = CODE_SPAN.sub("", line)
        got = INLINE_LINK.findall(line) + AUTOLINK.findall(line) + REFERENCE.findall(line)
        got += [a or b for a, b in ATTRIBUTE.findall(line)]
        found += [(number, link.strip("<>")) for link in got if link.strip("<>")]
    return found


@functools.lru_cache(maxsize=None)
def held(root):
    """The files the repository at `root` holds, as paths from it with forward slashes: what git tracks, or every
    file under `root` where it is no git checkout."""
    if (root / ".git").exists():
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
        return frozenset(out.decode("utf-8").split("\0")) - {""}
    return frozenset(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def sources(root):
    """[(file, is a page)]: the markdown files the repository holds, then the pages under its docs/."""
    files = sorted(held(root))
    return ([(f, False) for f in files if f.endswith(".md")]
            + [(f, True) for f in files if f.startswith("docs/") and f.endswith(".html")])


def read_from(base, path):
    """`path` read from the folder `base`, its . and .. parts resolved and its closing slash kept."""
    out = posixpath.normpath(posixpath.join(base, path.lstrip("/")))
    return out + "/" if path.endswith("/") else out


def named(link, source, page):
    """(path, served) for what `link`, held by the file `source`, names in the repository: the path from the
    repository's top with forward slashes, ending in a slash where the link names a folder, and `served`, true where
    the site answers the link and false where GitHub does. None for a link that leaves the repository, and for a
    fragment of the file itself."""
    url = urlsplit(link)
    path = unquote(url.path)
    host = url.hostname or ""
    if page and not url.scheme and not url.netloc and path.startswith("/"):
        host = SITE_HOST                         # a page's path from the root of the site's host
    elif url.scheme not in ("", "http", "https") or (url.netloc and not url.scheme):
        return None                              # mailto:, data: and the like, or another host
    if host == SITE_HOST:
        if path != SITE_PATH and not path.startswith(SITE_PATH + "/"):
            return None                          # another site of the same host
        return read_from("docs", path[len(SITE_PATH):] or "/"), True
    if host in (REPO_HOST, RAW_HOST):
        refs = r"/(?:blob|tree|raw|blame)/main/(.+)" if host == REPO_HOST else r"/main/(.+)"
        m = re.match(re.escape(REPO_PATH) + refs, path)
        return (read_from("", m.group(1)), False) if m else None
    if host or not path:
        return None
    return read_from("" if path.startswith("/") else posixpath.dirname(source), path), page


def holds(files, path, served):
    """True when `files`, the repository's, hold what a link to `path` opens, its name compared by case: the file,
    or for a folder its index.html on the site and the folder itself on GitHub. The site holds docs/ and nothing
    else."""
    name = path.rstrip("/")
    if name.startswith("..") or (served and name != "docs" and not name.startswith("docs/")):
        return False
    if name in files:
        return not path.endswith("/")
    if served:
        return f"{name}/index.html" in files
    return name == "." or any(f.startswith(name + "/") for f in files)


def targets(root, source, page):
    """[(line, link, path, served)] for every link of `source` that names something in the repository at `root`."""
    out = []
    for line, link in (page_links if page else markdown_links)(root / source):
        got = named(link, source, page)
        if got is not None:
            out.append((line, link, *got))
    return out


def dead_links(root):
    """Every link under `root` that names a file the repository does not hold, as `file:line: link`."""
    return [f"{source}:{line}: {link}"
            for source, page in sources(root)
            for line, link, path, served in targets(root, source, page)
            if not holds(held(root), path, served)]


def test_every_link_names_a_file_the_repository_holds():
    dead = dead_links(ROOT)
    assert not dead, "links to files the repository does not hold:\n" + "\n".join(dead)


def test_the_index_links_every_page_and_every_page_links_the_index():
    """docs/index.html is written by hand, so nothing lists a page for it: a page it leaves out is found here."""
    index = "docs/index.html"
    pages = [p for p, page in sources(ROOT) if page and p != index]
    assert pages, "no page was found under docs/"
    linked = {path for _, _, path, _ in targets(ROOT, index, True)}
    assert [p for p in pages if p not in linked] == []
    assert [p for p in pages if index not in {path for _, _, path, _ in targets(ROOT, p, True)}] == []


def test_the_site_is_docs_as_it_is():
    """The site's address of a page is its path under docs/, which is what this file resolves a link by: the pages
    workflow checks the repository out, uploads docs/ and deploys it, and runs nothing that could write or move a
    page."""
    workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert re.findall(r"^\s*path:\s*(\S+)", workflow, re.M) == ["docs"]
    assert not re.search(r"^\s*(?:-\s*)?run:", workflow, re.M)
    assert re.findall(r"uses:\s*([\w./-]+)@", workflow) == [
        "actions/checkout", "actions/upload-pages-artifact", "actions/deploy-pages"]


def test_a_link_to_a_file_the_repository_does_not_hold_is_reported(tmp_path):
    """On a small repository with one link of every kind held and one of every kind dead, exactly the dead ones are
    reported, each with its file and line."""
    (tmp_path / "docs" / "sub").mkdir(parents=True)
    (tmp_path / "docs" / "bare").mkdir()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("", encoding="utf-8")
    (tmp_path / "docs" / "bare" / "c.html").write_text("<p>no link</p>\n", encoding="utf-8")
    (tmp_path / "docs" / "index.html").write_text(
        '<a href="a.html">a</a> <a href="sub/b.html#part">b</a> <a href="sub/">sub</a>\n', encoding="utf-8")
    (tmp_path / "docs" / "sub" / "index.html").write_text('<a href="../index.html">up</a>\n', encoding="utf-8")
    (tmp_path / "docs" / "sub" / "b.html").write_text(
        '<a href="../a.html">a</a> <a href="/phi/a.html">a</a>\n<a href="/phi/gone.html">gone</a>\n', encoding="utf-8")
    (tmp_path / "docs" / "a.html").write_text(
        '<link rel="icon" href="data:,"><a href="index.html">index</a> <a href="#top">top</a>\n'
        '<a href="https://github.com/ezexe/phi">the repository</a> <a href="https://example.com/x.html">elsewhere</a>\n'
        '<script>var s = \'<a href="written-by-a-script.html">\';</script>\n'
        '<a href="A.html">another case</a>\n'
        '<a href="../README.md">outside docs/</a>\n'
        '<a href="bare/">a folder with no index.html</a>\n'
        '<img src="missing.png">\n', encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[a](docs/a.html), [the site](https://ezexe.github.io/phi/), [a page](https://ezexe.github.io/phi/sub/b.html),\n"
        "[code](https://github.com/ezexe/phi/blob/main/pkg/mod.py), [a folder](https://github.com/ezexe/phi/tree/main/pkg),\n"
        "[a folder](docs/bare), [mail](mailto:a@example.com), <https://example.com/>, `[in a code span](nowhere.md)`\n"
        "```\n[in a fenced block](nowhere.md)\n```\n"
        "[gone](docs/gone.html) [a file named as a folder](docs/a.html/)\n"
        "[the site under its builder](https://ezexe.github.io/phi/docs/a.html)\n"
        "[moved](https://github.com/ezexe/phi/blob/main/.docs/p4/nested-store.md)\n"
        '<a href="pkg/gone.py">raw HTML</a>\n'
        "[by reference][r]\n\n[r]: pkg/Mod.py\n"
        "[up and back](https://ezexe.github.io/phi/sub/../a.html) [raw](https://raw.githubusercontent.com/ezexe/phi/main/pkg/mod.py)\n"
        "[raw, gone](https://raw.githubusercontent.com/ezexe/phi/main/pkg/gone.py)\n", encoding="utf-8")
    assert dead_links(tmp_path) == [
        "README.md:7: docs/gone.html",
        "README.md:7: docs/a.html/",
        "README.md:8: https://ezexe.github.io/phi/docs/a.html",
        "README.md:9: https://github.com/ezexe/phi/blob/main/.docs/p4/nested-store.md",
        "README.md:10: pkg/gone.py",
        "README.md:13: pkg/Mod.py",
        "README.md:15: https://raw.githubusercontent.com/ezexe/phi/main/pkg/gone.py",
        "docs/a.html:4: A.html",
        "docs/a.html:5: ../README.md",
        "docs/a.html:6: bare/",
        "docs/a.html:7: missing.png",
        "docs/sub/b.html:2: /phi/gone.html",
    ]


def test_a_file_git_does_not_track_satisfies_no_link(tmp_path):
    """In a git checkout the repository holds what git tracks: a link to a file that is only on this machine is
    reported, and adding the file to git satisfies it."""
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "README.md").write_text("[notes](notes.txt)\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")
    git("add", "README.md")
    assert dead_links(tmp_path) == ["README.md:1: notes.txt"]
    git("add", "notes.txt")
    held.cache_clear()
    assert dead_links(tmp_path) == []
