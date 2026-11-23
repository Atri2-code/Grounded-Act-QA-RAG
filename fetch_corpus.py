"""Download the two Acts used in the evaluation into ./data.

Source: github.com/civictech-India/Indian-Law-Penal-Code-Json (bare Acts as JSON, derived from India Code).
The text is a snapshot: it may not reflect the latest amendments. Not legal advice.
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/civictech-India/Indian-Law-Penal-Code-Json/main/"
FILES = ["nia.json", "MVA.json"]


def main(dest: str = "data") -> None:
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = out / name
        with urllib.request.urlopen(BASE + name, timeout=60) as r:
            target.write_bytes(r.read())
        print(f"saved {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
