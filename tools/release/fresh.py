#!/usr/bin/env python3
"""Run the documented example in a newly created project outside the workspace."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="repository or unpacked release source")
    parser.add_argument("--packaged", action="store_true", help="patch registry edges to independently unpacked sibling .crate artifacts")
    args = parser.parse_args()
    source = args.source.resolve()
    with tempfile.TemporaryDirectory(prefix="alea-fresh-") as temporary:
        project = Path(temporary)
        (project / "src").mkdir()
        dependencies = "\n".join(
            f'{name} = {{ path = {json.dumps(str(source / "crates" / name))} }}'
            for name in ("alea-core", "alea-mcmc", "alea-runtime")
        )
        patches = ""
        if args.packaged:
            patches = "\n[patch.crates-io]\n" + "\n".join(
                f'{p.name} = {{ path = {json.dumps(str(p))} }}'
                for p in sorted((source / "crates").iterdir()) if (p / "Cargo.toml").is_file()
            ) + "\n"
        (project / "Cargo.toml").write_text(
            '[package]\nname = "alea-fresh-example"\nversion = "0.0.0"\n'
            'edition = "2024"\nrust-version = "1.95"\npublish = false\n'
            '[workspace]\n[dependencies]\n' + dependencies + "\n" + patches
        )
        (project / "src/main.rs").write_bytes((source / "tools/release/examples/reference.rs").read_bytes())
        env = dict(os.environ)
        # A fresh manifest/lock resolves independently; reuse only compiled objects.
        env.setdefault("CARGO_TARGET_DIR", str(source / "target/m17-fresh"))
        subprocess.run(["cargo", "generate-lockfile", "--offline"], cwd=project, env=env, check=True)
        subprocess.run(["cargo", "run", "--locked", "--offline"], cwd=project, env=env, check=True)


if __name__ == "__main__":
    main()
