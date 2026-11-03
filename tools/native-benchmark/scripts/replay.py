#!/usr/bin/env python3
"""Rebuild a summary to a NEW file without altering historical journals."""
import argparse
import json
from pathlib import Path

import bench
import compare
import kernel_compare


def journal(path):
    """Only a torn final record is recoverable; malformed complete rows fail."""
    if not path.exists():
        return []
    # Only LF terminates a journal record. Never skip malformed interior lines.
    lines = path.read_bytes().split(b'\n')
    return [bench.read_metrics(line) for line in lines[:-1]]


def rebuild(directory, kind):
    plan = json.loads((directory/('plan.json' if kind == 'paired' else 'manifest.json')).read_text())
    rows = journal(directory/'records.jsonl')
    def unique(records, keys):
        identities = [tuple(r[k] for k in keys) for r in records]
        if len(identities) != len(set(identities)):
            raise ValueError('duplicate journal identity')
    keys = ['case','engine','rep'] + (['mode'] if kind == 'paired' else ['size'] if kind == 'kernel' else [])
    unique(rows, keys)
    if kind == 'stan':
        return bench.summarize(rows,plan)
    decisions = journal(directory/'validation.jsonl')
    unique(decisions, ['case','rep', 'mode' if kind == 'paired' else 'size'])
    if kind == 'paired':
        return compare.summarize(rows,plan,decisions)
    return kernel_compare.summarize(rows,plan['cases'],decisions,plan['reps'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['stan','paired','kernel'])
    parser.add_argument('directory',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    result = rebuild(args.directory,args.kind)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
