"""Assemble the static demo site for GitHub Pages: the app's pages in demo mode, plus demo/.

    python scripts/build_site.py     -> site/

Standard library only, so the Pages workflow needs no installs. The pages are the app's own,
marked <html data-demo>: in demo mode they read demo/*.json instead of calling the server
(common.js has the switch), so the demo can't drift from the app.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"

shutil.rmtree(SITE, ignore_errors=True)
shutil.copytree(ROOT / "app" / "static", SITE, ignore=shutil.ignore_patterns(".gitkeep"))
shutil.copytree(ROOT / "demo", SITE / "demo")
for page in SITE.glob("*.html"):
    text = page.read_text(encoding="utf-8")
    if text.count('<html lang="en">') != 1:
        raise SystemExit(f'{page.name}: expected one <html lang="en"> tag to mark as demo')
    page.write_text(text.replace('<html lang="en">', '<html lang="en" data-demo>'), "utf-8")
print(f"built {SITE.relative_to(ROOT)}: {sorted(p.name for p in SITE.iterdir())}")
