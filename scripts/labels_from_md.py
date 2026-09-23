"""Generate data/labels.csv from the Labels table in data/labeling_sheet.md.

The sheet is the hand-written record and the only source of truth; the CSV is derived
from it. Re-run this after every relabel instead of editing the CSV by hand.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "data" / "labeling_sheet.md"
OUT = ROOT / "data" / "labels.csv"

# All 20 clips have now been filmed and labeled. Kept as a constant so that rows for
# clips that exist in the sheet but were never filmed are skipped rather than exported.
LAST_FILMED_CLIP = 20

# Reps where the push-up failed and the body never returned to the top position.
# A rep counts only once the person comes back up, so by that definition these are
# not reps at all and are dropped from the dataset rather than labeled as faults.
# Listed explicitly so the exclusions are visible and auditable, not silent.
EXCLUDED_REPS: set[tuple[str, int]] = {
    ("clip03", 8),
    ("clip08", 8),
    ("clip11", 6),
}

FAULTS = ["shallow", "hip_sag", "hip_pike", "no_lockout"]
OUT_COLUMNS = ["clip", "rep", *FAULTS, "notes"]


def split_row(line: str) -> list[str]:
    """Split one Markdown table line into its cell values.

    Backslashes are dropped because the sheet escapes underscores (``hip\\_sag``) to
    stop Markdown reading them as italics.
    """
    cells = line.strip().strip("|").split("|")
    return [cell.strip().replace("\\", "") for cell in cells]


def is_separator(cells: list[str]) -> bool:
    """True for the ``| :---- | :---- |`` line under a table header."""
    return all(re.fullmatch(r":?-+:?", cell) for cell in cells)


def read_label_rows(sheet: Path) -> list[dict[str, str]]:
    """Return every row of the Labels table as a dict keyed by column name."""
    lines = sheet.read_text(encoding="utf-8").splitlines()

    # The sheet holds two tables. Start at the "Labels" heading so the column-guide
    # table above it is never picked up by mistake.
    start = next(i for i, line in enumerate(lines) if line.lstrip("#* ").startswith("Labels"))

    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in lines[start:]:
        if not line.startswith("|"):
            if header is not None:
                break  # past the end of the table
            continue
        cells = split_row(line)
        if is_separator(cells):
            continue
        if header is None:
            header = cells
            continue
        # strict=True turns a row with a missing or extra cell into an error
        # instead of silently dropping a label.
        rows.append(dict(zip(header, cells, strict=True)))
    return rows


def was_filmed(clip: str) -> bool:
    return int(clip.removeprefix("clip")) <= LAST_FILMED_CLIP


def keep(row: dict[str, str]) -> bool:
    clip = row["clip"]
    return was_filmed(clip) and (clip, int(row["rep"])) not in EXCLUDED_REPS


def main() -> None:
    rows = [row for row in read_label_rows(SHEET) if keep(row)]

    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "clip": row["clip"],
                    "rep": row["rep"],
                    **{fault: row[fault] for fault in FAULTS},
                    "notes": row["comment"],
                }
            )

    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} rows")
    for fault in FAULTS:
        print(f"  {fault:<11} {sum(row[fault] == '1' for row in rows)}")
    clean = sum(all(row[fault] == "0" for fault in FAULTS) for row in rows)
    print(f"  {'clean':<11} {clean}")


if __name__ == "__main__":
    main()
