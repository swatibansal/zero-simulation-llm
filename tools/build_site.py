"""Generate docs/index.html (the GitHub Pages showcase) from zero_sim_demo.ipynb.

The executed notebook is the single source of truth: every code block, output panel
and figure on the page comes straight from it. Markdown rendering is a deliberate
subset — exactly the constructs the notebook uses — and raises on anything else so
content is never silently mangled.

Usage: python3 tools/build_site.py            (from the repo root)
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

REPO_URL = "https://github.com/swatibansal/zero-simulation-llm"
NOTEBOOK = "zero_sim_demo.ipynb"
OUTPUT = Path("docs/index.html")


# ---------------------------------------------------------------- markdown ----

def inline(text: str) -> str:
    """Escape HTML, then render `code`, **bold**, *italic* spans."""
    text = html.escape(text, quote=False)
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 2:
            out.append(f"<code>{part[1:-1]}</code>")
        else:
            part = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", part)
            part = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", part)
            out.append(part)
    return "".join(out)


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if not slug:
        raise ValueError(f"heading produced empty slug: {title!r}")
    return slug


def render_table(lines: list[str]) -> str:
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    if len(rows) < 2 or not all(re.fullmatch(r":?-{3,}:?", c) for c in rows[1] if c):
        raise ValueError(f"table without separator row: {lines[:2]}")
    head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>"
        for row in rows[2:]
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def markdown_to_html(source: str, headings: list[tuple[str, str]], drop_h1: bool = False) -> str:
    """Render the notebook's markdown subset. Appends (slug, title) of h2s to headings."""
    lines = source.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
        elif stripped.startswith("#"):
            m = re.match(r"^(#{1,3}) (.+)$", stripped)
            if not m:
                raise ValueError(f"unsupported heading: {stripped!r}")
            level = len(m.group(1))
            title = m.group(2)
            if level == 1 and drop_h1:
                pass  # the hero header replaces the notebook's h1
            elif level == 2:
                slug = slugify(title)
                headings.append((slug, title))
                out.append(f'<h2 id="{slug}">{inline(title)}</h2>')
            else:
                out.append(f"<h{level}>{inline(title)}</h{level}>")
            i += 1
        elif stripped.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(quote))}</p></blockquote>")
        elif stripped.startswith("|"):
            table = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table.append(lines[i])
                i += 1
            out.append(render_table(table))
        elif re.match(r"^[*-] ", stripped) or re.match(r"^\d+\. ", stripped):
            ordered = bool(re.match(r"^\d+\. ", stripped))
            marker = r"^\d+\. " if ordered else r"^[*-] "
            items: list[str] = []
            while i < len(lines):
                s = lines[i].strip()
                if re.match(marker, s):
                    items.append(re.sub(marker, "", s))
                elif s and lines[i].startswith("  "):  # continuation line
                    items[-1] += " " + s
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            body = "".join(f"<li>{inline(it)}</li>" for it in items)
            out.append(f"<{tag}>{body}</{tag}>")
        else:
            para = []
            while i < len(lines):
                s = lines[i].strip()
                if not s or re.match(r"^(#|>|\||[*-] |\d+\. )", s):
                    break
                para.append(s)
                i += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


# ------------------------------------------------------------------- cells ----

def render_code_cell(cell: dict) -> str:
    if not cell.get("execution_count"):
        raise ValueError("notebook has an unexecuted code cell — re-execute it first")
    src = html.escape("".join(cell["source"]), quote=False)
    parts = [
        '<div class="cell">',
        f'<div class="in"><span class="label">In [{cell["execution_count"]}]</span>'
        f'<pre><code class="language-python">{src}</code></pre></div>',
    ]
    for output in cell.get("outputs", []):
        kind = output["output_type"]
        if kind == "stream":
            text = html.escape("".join(output["text"]), quote=False)
            parts.append(
                f'<div class="out"><span class="label">Out</span><pre>{text}</pre></div>'
            )
        elif kind in ("display_data", "execute_result") and "image/png" in output.get("data", {}):
            png = "".join(output["data"]["image/png"]).replace("\n", "")
            parts.append(
                f'<figure><img src="data:image/png;base64,{png}" alt="simulation plot"></figure>'
            )
        elif kind == "execute_result":
            text = html.escape("".join(output["data"]["text/plain"]), quote=False)
            parts.append(
                f'<div class="out"><span class="label">Out</span><pre>{text}</pre></div>'
            )
        else:
            raise ValueError(f"unsupported output type: {kind}")
    parts.append("</div>")
    return "\n".join(parts)


# -------------------------------------------------------------------- page ----

STYLE = """
:root {
  --bg: #0b1220; --panel: #111a2c; --panel2: #060b14; --line: #223049; --line2: #2c3d5c;
  --ink: #d7e1ef; --bright: #f2f6fc; --muted: #8fa0b8;
  --accent: #5eb2ff; --green: #4ade80; --orange: #fb923c; --red: #f87171;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; color: var(--ink); background: var(--bg);
  font: 16px/1.7 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
::selection { background: rgba(94, 178, 255, .35); }
.wrap { display: grid; grid-template-columns: 240px minmax(0, 1fr); gap: 0 3rem;
  max-width: 1180px; margin: 0 auto; padding: 0 1.5rem; }
a { color: var(--accent); }

nav { position: sticky; top: 0; align-self: start; padding: 2rem 0; font-size: .84rem;
  max-height: 100vh; overflow-y: auto; }
nav a { display: block; color: var(--muted); text-decoration: none; padding: .3rem .75rem;
  border-left: 2px solid var(--line); transition: color .15s, border-color .15s; }
nav a:hover { color: var(--bright); }
nav a.active { color: var(--accent); border-left-color: var(--accent); font-weight: 600;
  text-shadow: 0 0 18px rgba(94, 178, 255, .55); }

header.hero { grid-column: 1 / -1; padding: 4rem 0 2.5rem; border-bottom: 1px solid var(--line);
  text-align: center;
  background:
    radial-gradient(ellipse 55% 90% at 50% -15%, rgba(94, 178, 255, .16), transparent),
    radial-gradient(ellipse 80% 70% at 50% -30%, rgba(74, 222, 128, .07), transparent); }
header.hero h1 { font-size: 2.1rem; line-height: 1.25; margin: 0 auto .6rem; color: var(--bright);
  letter-spacing: -.01em; max-width: 46rem; }
header.hero h1 span { color: var(--accent); }
header.hero p.tag { color: var(--muted); max-width: 50rem; margin: .3rem auto 1.3rem; font-size: 1.02rem; }
header.hero .links a { text-decoration: none; margin: 0 .75rem; font-weight: 600; font-size: .92rem; }
header.hero .links a:hover { text-decoration: underline; }
.claims { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;
  margin-top: 1.6rem; }
.claims div { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: .95rem 1.1rem; font-size: .88rem; color: var(--muted); line-height: 1.55;
  text-align: left; }
.claims strong { display: block; margin-bottom: .3rem; font-size: .95rem; }
.claims .c-blue   { border-top: 2px solid var(--accent); } .claims .c-blue strong   { color: var(--accent); }
.claims .c-green  { border-top: 2px solid var(--green); }  .claims .c-green strong  { color: var(--green); }
.claims .c-orange { border-top: 2px solid var(--orange); } .claims .c-orange strong { color: var(--orange); }

main { padding: 1.5rem 0 5rem; min-width: 0; }
main > p, main li, main blockquote { color: var(--ink); }
h2 { color: var(--bright); font-size: 1.5rem; margin: 3.5rem 0 1rem; padding-top: 2rem;
  border-top: 1px solid var(--line); scroll-margin-top: 1.25rem; }
h2::before { content: "// "; color: var(--accent); font-family: ui-monospace, monospace;
  font-size: 1.1rem; }
h3 { color: var(--bright); margin-top: 2.2rem; }
strong { color: var(--bright); }
blockquote { margin: 1.3rem 0; padding: .8rem 1.2rem; border-left: 3px solid var(--accent);
  background: var(--panel); border-radius: 0 8px 8px 0; }
blockquote p { margin: 0; }

table { border-collapse: collapse; margin: 1.2rem 0; font-size: .84rem; display: block;
  overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: .45rem .7rem; text-align: left; }
th { background: var(--panel); color: var(--bright); white-space: nowrap; }
tbody tr:nth-child(even) { background: rgba(17, 26, 44, .5); }
p code, li code, td code, th code, blockquote code, h2 code, h3 code {
  background: var(--panel); border: 1px solid var(--line); border-radius: 5px;
  padding: .08rem .35rem; font-size: .84em; color: #9ecbff; }

.cell { margin: 1.6rem 0; }
.cell .label { display: inline-block; font: 600 .68rem/1 ui-monospace, monospace;
  letter-spacing: .08em; text-transform: uppercase; padding: .25rem .6rem;
  border-radius: 6px 6px 0 0; position: relative; top: 1px; }
.cell .in .label { color: var(--accent); background: var(--panel); border: 1px solid var(--line);
  border-bottom: none; }
.cell .out .label { color: var(--green); background: var(--panel2); border: 1px solid var(--line);
  border-bottom: none; }
.cell .in pre { margin: 0; padding: 1rem 1.15rem; border: 1px solid var(--line);
  border-radius: 0 10px 10px 10px; overflow-x: auto; font-size: .8rem; line-height: 1.55;
  background: var(--panel); }
.cell .in pre code { background: transparent; padding: 0; }
.cell .out { margin-top: .8rem; }
.cell .out pre { margin: 0; padding: 1rem 1.15rem; border: 1px solid var(--line);
  border-radius: 0 10px 10px 10px; overflow-x: auto; font-size: .78rem; line-height: 1.55;
  background: var(--panel2); color: #b7e1c3; }

figure { margin: 1.3rem 0; text-align: center; }
figure img { max-width: 100%; background: #fff; padding: 10px; border-radius: 12px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, .45); }

footer { grid-column: 1 / -1; border-top: 1px solid var(--line); color: var(--muted);
  font-size: .85rem; padding: 1.75rem 0 3.5rem; }
@media (max-width: 900px) {
  .wrap { grid-template-columns: 1fr; gap: 0; }
  nav { position: static; max-height: none; padding-bottom: 0; }
  header.hero { padding-top: 2.5rem; }
}
"""

SCRIPT = """
const links = [...document.querySelectorAll('nav a')];
const byId = Object.fromEntries(links.map(a => [a.getAttribute('href').slice(1), a]));
const observer = new IntersectionObserver(entries => {
  for (const e of entries) if (e.isIntersecting) {
    links.forEach(a => a.classList.remove('active'));
    byId[e.target.id]?.classList.add('active');
  }
}, { rootMargin: '0px 0px -75% 0px' });
document.querySelectorAll('main h2').forEach(h => observer.observe(h));
"""

HERO = f"""
<header class="hero">
  <h1>zero-sim: <span>ZeRO-1 / ZeRO-2 / ZeRO-3</span> on 32 virtual GPUs</h1>
  <p class="tag">32 CPU threads pretend to be 32 GPUs (4 nodes × 8). One small model is trained four
  ways — plain data parallel, ZeRO-1, ZeRO-2, ZeRO-3 — while every byte in "HBM" and every byte on
  the "wire" is counted. Every code block below is the real simulation code, followed by the output
  it actually produced.</p>
  <div class="links">
    <a href="{REPO_URL}">GitHub repository →</a>
    <a href="{REPO_URL}/blob/main/zero_sim_demo.ipynb">Executed notebook →</a>
  </div>
  <div class="claims">
    <div class="c-blue"><strong>Same math, bit for bit</strong>All four strategies produce
    bitwise-identical weights after every step — sharding changes where bytes live, never the
    result.</div>
    <div class="c-green"><strong>Memory: 16Ψ → 16Ψ/N</strong>Measured per-GPU memory matches the
    ZeRO paper's formulas exactly, from full replication down to fully sharded.</div>
    <div class="c-orange"><strong>Traffic: 2Ψ vs 3Ψ</strong>ZeRO-1/2 move the same bytes as plain
    data parallel; ZeRO-3 pays 1.5× for gathering parameters just in time.</div>
  </div>
</header>
"""


def build(notebook_path: str = NOTEBOOK) -> str:
    nb = json.loads(Path(notebook_path).read_text())
    headings: list[tuple[str, str]] = []
    body: list[str] = []
    first_markdown = True
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            body.append(markdown_to_html("".join(cell["source"]), headings, drop_h1=first_markdown))
            first_markdown = False
        elif cell["cell_type"] == "code":
            body.append(render_code_cell(cell))
        else:
            raise ValueError(f"unsupported cell type: {cell['cell_type']}")

    nav = "\n".join(f'<a href="#{slug}">{inline(title)}</a>' for slug, title in headings)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>zero-sim: ZeRO-1/2/3 on 32 virtual GPUs</title>
<meta name="description" content="A pure-NumPy simulator of DP vs ZeRO-1/2/3 on 32 virtual GPUs: every simulation's code, output and plots.">
<meta name="theme-color" content="#0b1220">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<style>{STYLE}</style>
</head>
<body>
<div class="wrap">
{HERO}
<nav>{nav}</nav>
<main>
{chr(10).join(body)}
</main>
<footer>Generated from <code>zero_sim_demo.ipynb</code> by <code>tools/build_site.py</code> —
every output above was produced by executing the notebook top to bottom.</footer>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>hljs.highlightAll();</script>
<script>{SCRIPT}</script>
</body>
</html>
"""


def main() -> None:
    page = build()
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(page)
    (OUTPUT.parent / ".nojekyll").touch()
    print(f"wrote {OUTPUT} ({len(page) / 2**20:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
