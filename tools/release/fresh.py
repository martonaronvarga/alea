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
    parser.add_argument("--offline", action="store_true", help="require the independently resolved consumer dependencies to be cached")
    parser.add_argument("--workflow", action="store_true", help="exercise native models, diagnostics and durable recovery")
    parser.add_argument("--output", type=Path, help="new workflow output directory; default is temporary")
    args = parser.parse_args()
    source = args.source.resolve()
    with tempfile.TemporaryDirectory(prefix="alea-fresh-") as temporary:
        project = Path(temporary)
        (project / "src").mkdir()
        dependencies = "\n".join(
            f'{name} = {{ path = {json.dumps(str(source / "crates" / name))} }}'
            for name in ("alea",)
        )
        if args.workflow:
            dependencies += '\nserde_json = "1"\ntokio-util = { version = "0.7", features = ["rt"] }'
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
        example = "workflow.rs" if args.workflow else "reference.rs"
        (project / "src/main.rs").write_bytes((source / "tools/release/examples" / example).read_bytes())
        env = dict(os.environ)
        # A fresh manifest/lock resolves independently; reuse only compiled objects.
        env.setdefault("CARGO_TARGET_DIR", str(source / "target/fresh"))
        offline = ["--offline"] if args.offline else []
        subprocess.run(["cargo", "generate-lockfile", *offline], cwd=project, env=env, check=True)
        # A workspace fetch need not contain the independently resolved graph,
        # including packages selected by a different host's feature unification.
        subprocess.run(["cargo", "fetch", "--locked", *offline], cwd=project, env=env, check=True)
        command = ["cargo", "run", "--locked", "--offline"]
        if args.workflow:
            command += ["--", str(args.output.resolve() if args.output else project / "output")]
        subprocess.run(command, cwd=project, env=env, check=True)


if __name__ == "__main__":
    main()
