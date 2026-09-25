"""Build the final submission archive in the layout required by the challenge:

    <team>_submission.zip
    ├── output/matching_results.tsv
    ├── output/candidate_pairs.tsv
    ├── code/business_entity_resolution/{src/, scripts/, README.md, requirements.txt}
    └── Documentation_template.md

    python scripts/make_submission_zip.py --team myteam --out-dir output \
        --doc /path/to/filled/Documentation_template.md --dest .

requirements.txt inside the zip is pinned to the versions installed in the environment that runs
this script (i.e. the one that produced the outputs).
"""
from __future__ import annotations

import argparse
import os
import zipfile
from importlib import metadata

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "rapidfuzz", "faiss-cpu", "joblib",
            "jellyfish"]


def pinned_requirements() -> str:
    lines = []
    for p in PACKAGES:
        try:
            lines.append(f"{p}=={metadata.version(p)}")
        except metadata.PackageNotFoundError:
            if p != "jellyfish":  # optional
                lines.append(p)
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--doc", default=os.path.join(REPO, "Documentation_template.md"))
    ap.add_argument("--dest", default=".")
    a = ap.parse_args()

    zpath = os.path.join(a.dest, f"{a.team}_submission.zip")
    base = "code/business_entity_resolution"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            src = os.path.join(a.out_dir, f)
            if not os.path.exists(src):
                raise SystemExit(f"missing {src}")
            z.write(src, f"output/{f}")
        for folder in ("src", "scripts"):
            for root, _, files in os.walk(os.path.join(REPO, folder)):
                for f in files:
                    if f.endswith(".py"):
                        full = os.path.join(root, f)
                        z.write(full, f"{base}/{os.path.relpath(full, REPO)}")
        z.write(os.path.join(REPO, "README.md"), f"{base}/README.md")
        z.writestr(f"{base}/requirements.txt", pinned_requirements())
        z.write(a.doc, "Documentation_template.md")
    print(f"wrote {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB)")
    with zipfile.ZipFile(zpath) as z:
        for n in sorted(z.namelist()):
            print("  ", n)


if __name__ == "__main__":
    main()
