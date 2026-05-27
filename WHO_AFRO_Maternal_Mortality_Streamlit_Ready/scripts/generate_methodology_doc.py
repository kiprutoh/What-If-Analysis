#!/usr/bin/env python3
"""Generate a Word (.docx) version of the technical methodology markdown."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    md_path = root / "docs" / "Technical_Methodology.md"
    docx_path = root / "docs" / "Technical_Methodology.docx"

    if not md_path.exists():
        raise FileNotFoundError(md_path)

    from docx import Document  # type: ignore

    doc = Document()
    lines = md_path.read_text(encoding="utf-8").splitlines()

    def add_heading(text: str, level: int) -> None:
        doc.add_heading(text.strip(), level=level)

    for line in lines:
        if not line.strip():
            doc.add_paragraph("")
            continue

        if line.startswith("### "):
            add_heading(line[4:], 3)
        elif line.startswith("## "):
            add_heading(line[3:], 2)
        elif line.startswith("# "):
            add_heading(line[2:], 1)
        elif line.startswith("- "):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
        elif line.startswith("```"):
            # code fences: render as a simple monospace paragraph block
            doc.add_paragraph(line.strip("`"), style="Intense Quote")
        else:
            doc.add_paragraph(line)

    doc.save(str(docx_path))
    print(f"Wrote {docx_path}")


if __name__ == "__main__":
    main()

