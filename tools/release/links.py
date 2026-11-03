#!/usr/bin/env python3
"""Check Markdown file-target closure of checkout or Git-free source manifest."""
import argparse
from pathlib import Path

from source import ROOT, source_files, check_source_links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path,nargs='?',default=ROOT)
    args = parser.parse_args()
    paths = source_files(args.root)
    check_source_links({p.relative_to(args.root).as_posix(): p.read_bytes() for p in paths})
    print(f'source link closure: {len(paths)} files')


if __name__ == '__main__':
    main()
