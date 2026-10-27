#!/usr/bin/env python3
"""Validate denominators and retained failures before aggregating support shards."""
import argparse,json
from collections import Counter
from pathlib import Path


def aggregate(paths):
    seen=set(); counts=Counter(); processed=0; expected=None; profile=None; total=None
    for path in map(Path,paths):
        row=json.loads(path.read_text())
        shard,n=row['shard'],row['shards']
        if n<=0 or not 0<=shard<n or shard in seen:raise ValueError('invalid or duplicate shard')
        if expected is None:expected,profile,total=n,row['profile'],row['global_cases']
        if (n,row['profile'],row['global_cases'])!=(expected,profile,total):raise ValueError('mixed grid definitions')
        if profile=='full' and total!=42611184:raise ValueError('incorrect full-grid denominator')
        assigned=total//n+int(shard<total%n)
        if row['assigned']!=assigned or not 0<=row['processed']<=assigned:raise ValueError('incorrect shard denominator')
        complete=row['processed']==assigned
        if row['complete']!=complete:raise ValueError('incorrect completeness flag')
        for kind in ['density','fused']:
            if sum(row['counts'].get(f'{kind}_{status}',0) for status in ['finite','error'])!=row['processed']:raise ValueError('missing evaluation outcome')
        failures=Counter(); keys=set()
        with path.with_name('failures.jsonl').open() as stream:
            for line in stream:
                failed=json.loads(line); key=failed['index'],failed['kind']
                if key in keys or not 0<=key[0]<total or key[0]%n!=shard or key[1] not in ['density','fused']:raise ValueError('invalid or duplicate failure index')
                # Limited shards are the first processed members, not a subsample.
                if (key[0]-shard)//n>=row['processed']:raise ValueError('failure outside processed prefix')
                keys.add(key); failures[f'{key[1]}_error']+=1
        for kind in ['density','fused']:
            if failures[f'{kind}_error']!=row['counts'].get(f'{kind}_error',0):raise ValueError('missing retained failure')
        seen.add(shard);counts.update(row['counts']);processed+=row['processed']
    if expected is None:raise ValueError('no summaries')
    complete=len(seen)==expected and processed==total
    return dict(profile=profile,expected_cases=total,processed=processed,shards=len(seen),expected_shards=expected,complete=complete,counts=dict(counts),verdict='incomplete' if not complete else 'finite-output failures' if any(counts[f'{k}_error'] for k in ['density','fused']) else 'finite-output pass',scope='coverage only; finite outputs are not an accuracy certificate')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('summaries',nargs='+');args=parser.parse_args()
    print(json.dumps(aggregate(args.summaries),indent=2))
