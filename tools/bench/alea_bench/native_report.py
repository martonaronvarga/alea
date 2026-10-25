"""Backend-aware descriptive comparisons; undefined/failed pairs remain visible."""
import csv
import json
from pathlib import Path

from .manifest import write_new


def declared_backend(native, engine):
    if engine in ('alea','nuts-rs'):
        return 'native Rust / Enzyme; ModelBuilder source export' if native['rust_backend']=='enzyme' else 'BridgeStan / Stan AD'
    if engine=='blackjax':return 'native JAX translation'
    if engine=='stan':return 'native CmdStan 2.38.0 reverse AD'
    raise ValueError('unknown native experiment engine')


def comparison(report):
    rows = {(r['case'], r['method'], r['repetition']):r for r in report['rows']}
    result = []
    for (case, method, rep), alea in rows.items():
        if method not in ('nuts_cov_diag','nuts_cov_dense','nuts_grad_diag','nuts_grad_diag_jitter','nuts_grad_diag_position'): continue
        for competitor in ('nuts_rs','stan_nuts_diag','stan_nuts_dense','blackjax_nuts'):
            other = rows.get((case,competitor,rep))
            if other is None: continue
            row = dict(case=case,repetition=rep,alea_method=method,competitor=competitor,
                       alea_status=alea['status'],competitor_status=other['status'],
                       alea_backend=alea['backend'],competitor_backend=other['backend'],
                       comparison='sampling and adaptation workflow; library-specific tuning',
                       alea_rhat=(alea['diagnostics'] or {}).get('maximum_rhat'),
                       competitor_rhat=(other['diagnostics'] or {}).get('maximum_rhat'),
                       alea_divergences=alea['divergences'],competitor_divergences=other['divergences'])
            for key in ('process_seconds','warmup_seconds','retained_seconds',
                        'minimum_bulk_ess_per_process_second','minimum_bulk_ess_per_retained_second'):
                a, b = alea[key], other[key]
                defined = alea['status']==other['status']=='completed' and a is not None and b is not None and b>0
                row['alea_over_competitor_'+key] = a/b if defined else None
            row['undefined_reason'] = None if alea['status']==other['status']=='completed' else 'one or both jobs did not complete; retained in comparison inventory'
            result.append(row)
    return result


def write(directory):
    directory = Path(directory)
    report = json.loads((directory/'run/report.json').read_text())
    manifest = json.loads((directory/'run/manifest.json').read_text())
    pairs = comparison(report)
    write_new(directory/'comparisons.json', dict(purpose='descriptive',ratios='Alea / competitor',
        no_quality_filter=True,rows=pairs,notes=report['notes']))
    if pairs:
        with (directory/'comparisons.csv').open('x',newline='') as output:
            writer = csv.DictWriter(output,list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    prepared = manifest['preparation']
    write_new(directory/'preparation-cost.json', dict(
        native_worker=prepared['build_records'].get('native_worker'),
        stan_models=prepared['build_records'].get('records'),
        scope='Native build is batch/incremental; Stan records are reused historical builds. JIT is per measured process. These are explicit accounting scopes, not comparable per-model cold build times.',
        conformance=prepared['native']['conformance'],execution=prepared['native']['execution'],
        rust_backend=prepared['native']['rust_backend']))
    return pairs
