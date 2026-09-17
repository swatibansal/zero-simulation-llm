"""The generated Pages site must faithfully mirror the executed notebook."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from build_site import build  # noqa: E402

NB = json.loads((ROOT / "zero_sim_demo.ipynb").read_text())


def _page() -> str:
    return build(str(ROOT / "zero_sim_demo.ipynb"))


def test_one_section_per_notebook_heading():
    md = "\n".join("".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "markdown")
    want = re.findall(r"(?m)^## (.+)$", md)
    got = re.findall(r"<h2 id=", _page())
    assert len(got) == len(want) == 10


def test_one_code_block_and_all_outputs_per_code_cell():
    code_cells = [c for c in NB["cells"] if c["cell_type"] == "code"]
    images = sum(
        1 for c in code_cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})
    )
    page = _page()
    assert page.count('class="language-python"') == len(code_cells) == 17
    assert page.count("data:image/png;base64,") == images == 7


def test_notebook_output_text_appears_verbatim():
    # a stable line from the bitwise-identity check must survive rendering
    page = _page()
    assert "max |Δweight| vs DP after 12 steps = 0.0" in page


def test_no_course_specific_strings():
    assert not re.search(r"(?i)era v5|cowork|session guide", _page())


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
