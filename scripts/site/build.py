"""
Build the phi docs site: every Markdown doc, notebook, interactive component (.jsx) and saved artifact in the repo,
rendered to static HTML for GitHub Pages (.github/workflows/pages.yml).

  README.md, CONTRIBUTING.md         -> readme.html, contributing.html
  .claude/skills/p4-model/SKILL.md   -> skills/p4-model.html
  .docs/**/*.md                      -> docs/**/*.html
  .docs/**/*.ipynb                   -> docs/**/*.html (nbconvert, outputs as saved)
  .docs/**/*.jsx                     -> docs/**/*.jsx.html (React 18 + Babel in the page)
  .docs/artifacts/*.html             -> docs/artifacts/*.html (a page fragment gets a document skeleton)
  every other file under .docs/      -> copied as is, so the docs' links to results/ and data resolve

A relative link to a file the site doesn't carry goes to the file on GitHub instead.

    pip install markdown nbconvert
    python scripts/site/build.py --out _site
"""

import argparse
import html
import os
import posixpath
import re
import shutil
import sys

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "https://github.com/ezexe/phi"
BRANCH = "main"
REACT = "https://cdnjs.cloudflare.com/ajax/libs/react/18.3.1/umd/react.production.min.js"
REACT_DOM = "https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.3.1/umd/react-dom.production.min.js"
BABEL = "https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.24.7/babel.min.js"
TOP = {"README.md": "readme.html", "CONTRIBUTING.md": "contributing.html",
       ".claude/skills/p4-model/SKILL.md": "skills/p4-model.html"}
SECTIONS = [("docs/phi/", "φ — the register and its matrix"), ("docs/pho/", "pho — theory and tiers"),
            ("docs/p4/", "p4 — the nested store"), ("docs/artifacts/", "Artifacts"), ("docs/", "Walkthroughs"),
            ("", "The repository")]

STYLE = """
:root { --bg: #f2f4f7; --panel: #ffffff; --ink: #172033; --muted: #5a6477; --line: #cdd4df; --one: #a8661a;
  --zero: #2f679f; --code: #e9edf3;
  --display: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif;
  --body: "Instrument Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; color-scheme: light }
@media (prefers-color-scheme: dark) { :root { --bg: #0e131b; --panel: #151c27; --ink: #e3e8ef; --muted: #9aa5b7;
  --line: #2c3748; --one: #e2a556; --zero: #7eaee0; --code: #1b2330; color-scheme: dark } }
* { box-sizing: border-box }
body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.65 var(--body) }
.bar { border-bottom: 1px solid var(--line); background: var(--panel) }
.bar div, main { max-width: 860px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)) }
.bar div { display: flex; gap: 16px; flex-wrap: wrap; align-items: baseline; padding-block: 12px }
.bar a { color: var(--ink); text-decoration: none; font: 600 0.9rem var(--mono) }
.bar a.home { color: var(--one) }
.bar .src { margin-left: auto; color: var(--muted); font-weight: 400 }
main { padding-block: 32px 64px }
main > * { max-width: 72ch }
main > table, main > pre, main > .tbl { max-width: none }
h1, h2, h3, h4 { font-family: var(--display); line-height: 1.2; text-wrap: balance }
h1 { font-size: clamp(1.8rem, 4.5vw, 2.5rem); margin: 0 0 16px }
a { color: var(--zero) }
code, pre { font-family: var(--mono); font-size: 0.88em }
code { background: var(--code); padding: 0.1em 0.35em; border-radius: 3px }
pre { background: var(--code); padding: 14px 16px; border-radius: 6px; overflow-x: auto; line-height: 1.5 }
pre code { background: none; padding: 0 }
.tbl { overflow-x: auto }
table { border-collapse: collapse; font-size: 0.93rem; font-variant-numeric: tabular-nums }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top }
th { color: var(--muted); font-weight: 600 }
img { max-width: 100% }
blockquote { margin: 0; padding-left: 14px; border-left: 3px solid var(--line); color: var(--muted) }
.index h2 { font-size: 1.1rem; margin: 32px 0 8px; color: var(--muted) }
.index ul { list-style: none; padding: 0; margin: 0; display: grid; gap: 2px }
.index li a { display: flex; gap: 12px; flex-wrap: wrap; padding: 8px 0; border-bottom: 1px solid var(--line);
  text-decoration: none; color: var(--ink) }
.index li span { color: var(--muted); font: 0.8rem var(--mono); margin-left: auto }
"""
FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,'
         '600&family=Instrument+Sans:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">')


def page(title, body, depth, src=None, wide=False):
    up = "../" * depth
    src_link = f'<a class="src" href="{REPO}/blob/{BRANCH}/{src}">{html.escape(src)}</a>' if src else ""
    main = body if wide else f"<main>{body}</main>"
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} · phi</title>'
            f'{FONTS}<style>{STYLE}</style></head><body>'
            f'<nav class="bar"><div><a class="home" href="{up}index.html">φ phi</a>{src_link}</div></nav>'
            f'{main}</body></html>\n')


def collect():
    """{repo path: site path} for every page, and the .docs files copied as they are."""
    pages, copies = {}, []
    for src, out in TOP.items():
        if os.path.exists(os.path.join(ROOT, src)):
            pages[src] = out
    for dirpath, dirs, files in os.walk(os.path.join(ROOT, ".docs")):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "__")))
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(dirpath, f), ROOT).replace(os.sep, "/")
            site = "docs/" + rel[len(".docs/"):]
            stem, ext = os.path.splitext(site)
            if f.endswith((".pyc",)):
                continue
            if ext in (".md", ".ipynb"):
                pages[rel] = stem + ".html"
            elif ext == ".jsx":
                pages[rel] = site + ".html"
                copies.append((rel, site))
            elif ext == ".html" and rel.startswith(".docs/artifacts/"):
                pages[rel] = site
            else:
                copies.append((rel, site))
    return pages, copies


def rewrite_links(body, src, out, pages, copied):
    """Relative links: to the page or copy the site carries, else to the file on GitHub."""
    src_dir, out_dir = posixpath.dirname(src), posixpath.dirname(out)

    def fix(m):
        attr, q, url = m.group(1), m.group(2), m.group(3)
        if re.match(r"^([a-z][a-z0-9+.-]*:|#|//)", url, re.I) or not url:
            return m.group(0)
        path, frag = (url.split("#", 1) + [""])[:2]
        target = posixpath.normpath(posixpath.join(src_dir, path)) if path else src
        if target in pages:
            new = posixpath.relpath(pages[target], out_dir or ".")
        elif target in copied:
            new = posixpath.relpath(copied[target], out_dir or ".")
        elif not target.startswith(".."):
            kind = "tree" if os.path.isdir(os.path.join(ROOT, target)) else "blob"
            new = f"{REPO}/{kind}/{BRANCH}/{target}"
        else:
            return m.group(0)
        return f'{attr}={q}{new}{"#" + frag if frag else ""}{q}'
    return re.sub(r'(href|src)=(["\'])([^"\']*)\2', fix, body)


def render_md(src):
    with open(os.path.join(ROOT, src), encoding="utf-8") as f:
        text = f.read()
    if text.startswith("---\n"):                                   # a skill's front matter
        end = text.find("\n---\n", 4)
        text = text[end + 5:] if end > 0 else text
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "toc"])
    body = re.sub(r"<table>", '<div class="tbl"><table>', body).replace("</table>", "</table></div>")
    m = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    title = re.sub(r"<[^>]+>", "", m.group(1)) if m else os.path.basename(src)
    return html.unescape(title).strip(), body


def render_ipynb(src):
    from nbconvert import HTMLExporter
    body, _ = HTMLExporter(template_name="lab").from_filename(os.path.join(ROOT, src))
    return body


def render_jsx(src, out):
    with open(os.path.join(ROOT, src), encoding="utf-8") as f:
        code = f.read()
    if "</script" in code.lower():
        raise ValueError(f"{src} holds </script, which would end the inline source")
    name = posixpath.basename(src)
    m = re.search(r"^export\s+default\s+function\s+(\w+)", code, re.M)
    title = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", m.group(1)) if m else name
    body = (f'<div id="root" style="padding:16px">Loading {html.escape(name)}…</div>'
            f'<script id="src" type="text/plain">{code}</script>'
            f'<script src="{REACT}"></script><script src="{REACT_DOM}"></script><script src="{BABEL}"></script>'
            '<script>\n'
            'let src = document.getElementById("src").textContent;\n'
            'src = src.replace(/^import\\s*\\{([^}]+)\\}\\s*from\\s*["\']react["\'];?\\s*$/m, "const {$1} = React;");\n'
            'src = src.replace(/^export\\s+default\\s+/m, "window.__App = ");\n'
            'try {\n'
            f'  const {{ code }} = Babel.transform(src, {{ presets: ["react"], filename: {name!r} }});\n'
            '  (0, eval)(code);\n'
            '  ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(window.__App));\n'
            '} catch (e) { document.getElementById("root").textContent = "Could not render: " + e.message; }\n'
            '</script>')
    return title, body


def wrap_fragment(text, title):
    if re.search(r"<html[\s>]", text[:2000], re.I):
        return text
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            f'</head><body style="margin:0">\n{text}\n</body></html>\n')


def build(out):
    pages, copies = collect()
    copied = {src: site for src, site in copies}
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    open(os.path.join(out, ".nojekyll"), "w").close()
    for src, site in copies:
        dst = os.path.join(out, site)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, src), dst)
    listed = []
    for src, site in sorted(pages.items()):
        depth = site.count("/")
        if src.endswith(".md"):
            title, body = render_md(src)
            doc = page(title, rewrite_links(body, src, site, pages, copied), depth, src)
            kind = "doc"
        elif src.endswith(".ipynb"):
            title, doc, kind = posixpath.basename(src), render_ipynb(src), "notebook"
        elif src.endswith(".jsx"):
            title, body = render_jsx(src, site)
            doc, kind = page(title, body, depth, src, wide=True), "interactive"
        else:
            with open(os.path.join(ROOT, src), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r"<title>(.*?)</title>", text, re.S)
            title, kind = (html.unescape(m.group(1)).strip() if m else posixpath.basename(src)), "artifact"
            doc = wrap_fragment(text, title)
        dst = os.path.join(out, site)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(doc)
        listed.append((site, title, kind))
    groups = []
    taken = set()
    for prefix, name in SECTIONS:
        items = [x for x in listed if x[0] not in taken and x[0].startswith(prefix)
                 and (prefix or "/" not in x[0] or x[0].startswith("skills/"))]
        if prefix == "docs/":
            items = [x for x in items if "/" not in x[0][len(prefix):]]
        taken.update(x[0] for x in items)
        if items:
            lis = "".join(f'<li><a href="{s}">{html.escape(t)}<span>{k}</span></a></li>' for s, t, k in items)
            groups.append(f"<h2>{html.escape(name)}</h2><ul>{lis}</ul>")
    intro = ('<h1>phi</h1><p>φ-based Fibonacci-weighted encoding and φ-prune structured sparsity, and the p4 models '
             'built on them. Every doc, notebook, interactive component and artifact in '
             f'<a href="{REPO}">ezexe/phi</a>, rebuilt on every push to main.</p>')
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(page("Docs", f'<div class="index">{intro}{"".join(groups)}</div>', 0))
    missing = [x for x in listed if x[0] not in taken]
    return listed, missing


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "_site"))
    args = ap.parse_args(argv)
    listed, missing = build(args.out)
    print(f"site — {len(listed)} pages → {args.out}" + (f"; not indexed: {[m[0] for m in missing]}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
