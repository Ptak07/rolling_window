"""Convert man/*.Rd files to docs/python/r_api/*.rst for Sphinx."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MAN_DIR = ROOT / "man"
OUT_DIR = Path(__file__).parent / "python" / "r_api"


def _strip_rd(text: str) -> str:
    """Remove simple Rd markup: \\code{x} → ``x``, \\code{NA} → ``NA``, etc."""
    text = re.sub(r"\\code\{([^}]*)\}", r"``\1``", text)
    text = re.sub(r"\\pkg\{([^}]*)\}", r"**\1**", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", text)
    text = re.sub(r"\\strong\{([^}]*)\}", r"**\1**", text)
    text = re.sub(r"\\link\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)
    return text.strip()


def _extract(content: str, tag: str) -> str:
    """Extract the body of a \\tag{...} block (handles nested braces)."""
    pattern = f"\\{tag}{{"
    start = content.find(pattern)
    if start == -1:
        return ""
    idx = start + len(pattern)
    depth = 1
    chars: list[str] = []
    while idx < len(content) and depth > 0:
        ch = content[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth > 0:
            chars.append(ch)
        idx += 1
    return "".join(chars).strip()


def _extract_items(content: str, tag: str) -> list[tuple[str, str]]:
    """Extract all \\item{name}{desc} pairs inside a \\tag{} block."""
    block = _extract(content, tag)
    items: list[tuple[str, str]] = []
    pos = 0
    while True:
        m = re.search(r"\\item\s*\{", block[pos:])
        if not m:
            break
        abs_start = pos + m.start() + len(m.group())
        # extract first brace group (name)
        depth = 1
        name_chars: list[str] = []
        i = abs_start
        while i < len(block) and depth > 0:
            ch = block[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            if depth > 0:
                name_chars.append(ch)
            i += 1
        name = "".join(name_chars).strip()
        i += 1  # skip closing }
        # extract second brace group (description)
        while i < len(block) and block[i] in (" ", "\n", "\r", "\t"):
            i += 1
        desc_chars: list[str] = []
        if i < len(block) and block[i] == "{":
            i += 1
            depth = 1
            while i < len(block) and depth > 0:
                ch = block[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                if depth > 0:
                    desc_chars.append(ch)
                i += 1
        desc = "".join(desc_chars).strip()
        items.append((name, desc))
        pos = i + 1
    return items


def rd_to_rst(rd_path: Path) -> str:
    content = rd_path.read_text(encoding="utf-8")

    title = _strip_rd(_extract(content, "title"))
    description = _strip_rd(_extract(content, "description"))
    usage = _extract(content, "usage").strip()
    value = _strip_rd(_extract(content, "value"))
    examples_raw = _extract(content, "examples").strip()
    args = _extract_items(content, "arguments")

    rst = f"{title}\n{'=' * len(title)}\n\n"

    rst += f"{description}\n\n"

    if usage:
        rst += "Usage\n-----\n\n"
        rst += f".. code-block:: r\n\n"
        for line in usage.splitlines():
            rst += f"   {line}\n"
        rst += "\n"

    if args:
        rst += "Parameters\n----------\n\n"
        rst += ".. list-table::\n"
        rst += "   :header-rows: 1\n"
        rst += "   :widths: 20 80\n\n"
        rst += "   * - Parameter\n"
        rst += "     - Description\n"
        for name, desc in args:
            cleaned = _strip_rd(desc).replace("\n", " ")
            rst += f"   * - ``{name}``\n"
            rst += f"     - {cleaned}\n"
        rst += "\n"

    if value:
        rst += "Returns\n-------\n\n"
        rst += f"{value}\n\n"

    if examples_raw:
        lines = [l for l in examples_raw.splitlines() if not l.strip().startswith("%")]
        example_code = "\n".join(lines).strip()
        if example_code:
            rst += "Examples\n--------\n\n"
            rst += ".. code-block:: r\n\n"
            for line in example_code.splitlines():
                rst += f"   {line}\n"
            rst += "\n"

    return rst


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rd_files = sorted(MAN_DIR.glob("*.Rd"))
    # skip package-level Rd
    rd_files = [f for f in rd_files if not f.stem.endswith("-package")]

    names: list[str] = []
    for rd in rd_files:
        rst_content = rd_to_rst(rd)
        out_path = OUT_DIR / f"{rd.stem}.rst"
        out_path.write_text(rst_content, encoding="utf-8")
        names.append(rd.stem)
        print(f"  {rd.name} → {out_path.relative_to(ROOT)}")

    index = "R API Reference\n===============\n\n"
    index += "All functions accept a numeric vector ``x`` (and ``y`` for bivariate\n"
    index += "functions), a ``window_size`` integer, and an optional ``min_periods``\n"
    index += "parameter compatible with *pandas* semantics.\n\n"
    index += ".. toctree::\n"
    index += "   :maxdepth: 1\n"
    index += "   :caption: Functions\n\n"
    for name in sorted(names):
        index += f"   {name}\n"
    (OUT_DIR / "index.rst").write_text(index, encoding="utf-8")
    print(f"  → r_api/index.rst ({len(names)} entries)")


if __name__ == "__main__":
    sys.exit(main())
