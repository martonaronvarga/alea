"""Reproduce descriptive workflow summaries from a complete sealed run."""
import argparse
import collections
import csv
import json
import statistics
from pathlib import Path


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value):
    return '—' if value is None else f'{value:.4g}'


def write_csv(path, rows, fields):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def analyze(run, output, legacy=None):
    report = json.loads((run / 'report.json').read_text())
    groups, rows = report['groups'], report['rows']
    manifest = json.loads((run / 'manifest.json').read_text())
    protocol = manifest['protocol']
    methods = protocol['methods']
    repetitions = protocol['repetitions']
    cases = sorted({g['case'] for g in groups})
    denominator = len(cases) * repetitions
    by = {(g['case'], g['method']): g for g in groups}
    native_jax = {
        'Rate_1_data-Rate_1_model', 'Rate_2_data-Rate_2_model',
        'eight_schools-eight_schools_centered',
        'eight_schools-eight_schools_noncentered',
        'wells_data-wells_dist100_model', 'kidiq-kidscore_momhs',
    }
    summary = report['summary']
    lines = [
        '# Native warmup and research workflow results', '',
        f"{summary['planned_jobs']} planned jobs: {summary['status_counts']}. "
        f"{len(cases)} posteriors, {len(methods)} configurations, {repetitions} repetitions, "
        f"{protocol['chains']} chains; {protocol['warmup']} warmup and {protocol['draws']} retained draws per chain.", '',
        f"Measured on one {report['host']['cpu']} host. Jobs run serially on one allowed CPU. "
        'Compilation and preparation precede measurement; fresh-process wall time uses nanosecond clocks and pidfd completion.', '',
        'Completion is a technical outcome. R-hat, ESS, posterior reference discrepancies, divergences and costs '
        'remain separate observations. Medians condition on completion with denominators shown. '
        'No promotion score, quality eligibility filter or default-selection decision is computed.', '',
        '## Configuration coverage', '',
        '| Configuration | Completed / planned | Failed | Unsupported | Worst R-hat | Smallest bulk ESS | Largest median max mean error / reference SD |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    coverage = []
    for method in methods:
        selected = [g for g in groups if g['method'] == method]
        defined = lambda key: [g[key] for g in selected if g[key] is not None]
        record = dict(method=method, planned=denominator,
                      completed=sum(g['completed'] for g in selected),
                      failed=sum(g['failed'] for g in selected),
                      unsupported=sum(g['unsupported'] for g in selected),
                      worst_rhat=max(defined('worst_rhat'), default=None),
                      smallest_bulk_ess=min(defined('smallest_bulk_ess'), default=None),
                      largest_median_mean_error=max(defined('median_maximum_abs_mean_error_reference_sd'), default=None))
        coverage.append(record)
        lines.append(f"| {method} | {record['completed']}/{denominator} | {record['failed']} | {record['unsupported']} | "
                     + ' | '.join(fmt(record[k]) for k in ('worst_rhat', 'smallest_bulk_ess', 'largest_median_mean_error')) + ' |')
    lines += ['', 'Worst values span different posteriors and budgets and do not define a ranking. '
              'RMHMC remains exactness-pending with its smaller 32+64 budget and dimension limit. '
              'MCLMC is approximate. ATLAS uses a full research proposal with deterministic curvature; '
              'Pathfinder rows include initialization plus native NUTS.', '',
              '## Baseline costs alongside statistical results', '',
              'Alea baseline means covariance-diagonal native NUTS warmup and frozen NUTS. '
              'Each ratio below uses per-posterior medians only where both workflows complete all declared repetitions. '
              'Counts state that availability condition; they do not indicate convergence. '
              'All cases and their diagnostic discrepancies remain in the full report.', '',
              '| Comparator | Available paired posteriors | Median Alea / comparator process time | Median Alea / comparator retained time | Median Alea / comparator bulk ESS per process second |',
              '| --- | ---: | ---: | ---: | ---: |']
    paired = []
    comparisons = []
    for method, label, subset in [
        ('stan_nuts_diag', 'CmdStan diagonal NUTS', set(cases)),
        ('stan_nuts_dense', 'CmdStan dense NUTS', set(cases)),
        ('nuts_rs', 'nuts-rs diagonal NUTS', set(cases)),
        ('blackjax_nuts', 'BlackJAX native JAX targets', native_jax),
        ('blackjax_nuts', 'BlackJAX BridgeStan callbacks', set(cases) - native_jax),
    ]:
        available = []
        for case in cases:
            a, b = by.get((case, 'nuts_cov_diag')), by.get((case, method))
            if case not in subset or a is None or b is None:
                continue
            record = dict(case=case, comparator=method, backend_group=label,
                          alea_completed=a['completed'], comparator_completed=b['completed'],
                          planned_repetitions=repetitions)
            for prefix, group in [('alea', a), ('comparator', b)]:
                for key in ('worst_rhat', 'smallest_bulk_ess', 'median_maximum_abs_mean_error_reference_sd',
                            'median_maximum_abs_sd_ratio_error', 'median_process_seconds', 'median_retained_seconds'):
                    record[prefix + '_' + key] = group[key]
            both_complete = a['completed'] == repetitions and b['completed'] == repetitions
            record['both_complete'] = both_complete
            for name, key in [('process_ratio', 'median_process_seconds'), ('retained_ratio', 'median_retained_seconds'),
                              ('ess_per_process_second_ratio', 'median_minimum_bulk_ess_per_process_second')]:
                record[name] = a[key] / b[key] if both_complete and a[key] is not None and b[key] is not None and b[key] > 0 else None
            paired.append(record)
            if both_complete:
                available.append(record)
        result = dict(comparator=method, backend_group=label, paired_posteriors=len(available),
                      **{key: median(r[key] for r in available) for key in ('process_ratio', 'retained_ratio', 'ess_per_process_second_ratio')})
        comparisons.append(result)
        lines.append(f"| {label} | {len(available)}/{len(set(cases) & subset)} | "
                     + ' | '.join(fmt(result[k]) for k in ('process_ratio', 'retained_ratio', 'ess_per_process_second_ratio')) + ' |')
    lines += ['', 'A time ratio below one indicates less recorded time for Alea. An ESS/s ratio above one indicates '
              'greater recorded bulk ESS per second in those completed comparisons. Inspect the paired R-hat and reference '
              'errors in [baseline-pairs.csv](baseline-pairs.csv) before interpreting either. '
              'No ratio supplies a scientific success gate. Native retained scopes differ: Stan includes CSV/generated quantities; '
              'the Rust worker includes owned retained positions and kernel conversion but excludes serialization. '
              'Cold process times include startup, initialization, adaptation, sampling and output. '
              'BlackJAX callback timings include host transfers and do not establish native-JAX speed.', '',
              '## Difficult starts', '',
              '| Posterior | Method | Completed / repetitions | Worst R-hat | Smallest bulk ESS | Median largest mean error / reference SD | Median largest spread-ratio error |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for case in ('earnings-earn_height', 'sblri-blr'):
        for method in ('nuts_cov_diag', 'nuts_cov_dense', 'nuts_fisher_diag', 'nuts_fisher_rank4', 'nuts_pathfinder_raw', 'nuts_pathfinder_psis', 'atlas_full', 'stan_nuts_diag', 'nuts_rs'):
            g = by.get((case, method))
            if g is None:
                continue
            lines.append(f"| {case} | {method} | {g['completed']}/{g['repetitions']} | "
                         + ' | '.join(fmt(g[k]) for k in ('worst_rhat', 'smallest_bulk_ess', 'median_maximum_abs_mean_error_reference_sd', 'median_maximum_abs_sd_ratio_error')) + ' |')
    lines += ['', 'The 128+64 smoke run and earlier diagnostic pilot are separate evidence. '
              'They are not pooled with this experiment. Empirical PosteriorDB reference summaries are not exact truth; '
              'the full coordinate file retains MCSE and reference spread for each comparison.', '',
              '## Failures', '']
    failures = []
    for row in rows:
        if row['status'] == 'completed':
            continue
        failures.append(dict(job=row['job'], case=row['case'], method=row['method'], status=row['status'],
                             reason=row['unsupported_reason'] or row['failure']['message']))
    for method in methods:
        selected = [r for r in failures if r['method'] == method]
        if not selected:
            continue
        counts = collections.Counter(r['status'] for r in selected)
        examples = list(dict.fromkeys(r['reason'] for r in selected))[:2]
        lines.append(f"- `{method}`: {dict(counts)}. Examples: " + '; '.join('`' + x.replace('`', '').replace('\n', ' ')[:210] + '`' for x in examples) + '.')
    lines += ['', 'All failures, timeouts, unsupported configurations and partial outputs remain in the denominator. '
              'A strict SoftAbs symmetry rejection is a backend-contract failure rather than an RMHMC inference timing. '
              'Finite-solver replay does not establish volume preservation. '
              'Finite-point score/output failures through frozen affine targets retain their declared typed error behavior. '
              'Do not replace these outcomes with retries or fallback policies after measurement.', '',
              '## Evidence', '',
              '[Interactive report](report.html), [all groups](summary.csv), [coordinate comparisons](reference-comparisons.csv), '
              '[phase costs](phase-costs.csv), [initialization paths](initialization-paths.csv), [failures](failures.csv). '
              'The complete bundle retains every result, raw draw, trace and source/preparation archive. '
              'Reproduce this supplementary analysis with `python3 analysis.py RUN_DIRECTORY OUTPUT_DIRECTORY`.', '',
              f"Manifest: `{report['manifest_sha256']}`."]
    write_csv(output/'coverage.csv', coverage, list(coverage[0]))
    write_csv(output/'baseline-pairs.csv', paired, list(paired[0]) if paired else ['case'])
    write_csv(output/'failures.csv', failures, ['job', 'case', 'method', 'status', 'reason'])
    (output/'paired-baseline-costs.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    (output/'findings.md').write_text('\n'.join(lines) + '\n')
    phases, initializations = [], []
    for row in rows:
        record = json.loads((run/'jobs'/row['job']/'result.json').read_text())
        worker = record['worker'] or {}
        worker = dict(worker, **worker.get('partial', {}))
        identity = dict(job=row['job'], case=row['case'], method=row['method'], repetition=row['repetition'], status=row['status'])
        for p in worker.get('phases', []):
            for phase in ('warmup', 'retained', 'serialization', 'initialization'):
                if phase + '_seconds' not in p:
                    continue
                start = p.get('work_before') if phase == 'warmup' else p.get('work_before_retained') if phase == 'retained' else None
                end = p.get('work_before_retained') if phase == 'warmup' else p.get('work_after') if phase == 'retained' else None
                work = {k: end[k] - start[k] if start and end and k in start and k in end else None
                        for k in ('gradient_api_calls', 'hessian_api_calls', 'hvp_api_calls', 'third_ad_api_calls')}
                phases.append(dict(identity, chain=p.get('chain', 'cohort'), phase=phase, seconds=p[phase+'_seconds'], **work))
            initializer = p.get('initialization')
            if initializer:
                for path in initializer.get('paths', []):
                    initializations.append(dict(identity, chain=p.get('chain', 'cohort'),
                                                importance_ess=initializer.get('importance_ess'), pareto_shape=initializer.get('pareto_shape'),
                                                psis_status=initializer.get('psis_status'), path=path['path'],
                                                path_status=path['status'], optimization_steps=path['optimization_steps'],
                                                target_evaluations=path['target_evaluations']))
    write_csv(output/'phase-costs.csv', phases, list(phases[0]) if phases else ['job'])
    write_csv(output/'initialization-paths.csv', initializations, list(initializations[0]) if initializations else ['job'])
    dispersion = []
    for g in groups:
        for status in ('completed', 'failed', 'unsupported'):
            selected = [r for r in rows if (r['case'], r['method'], r['status']) == (g['case'], g['method'], status)]
            if not selected:
                continue
            for phase in ('process', 'warmup', 'retained', 'initialization', 'serialization', 'jit'):
                values = [r[phase+'_seconds'] for r in selected if r[phase+'_seconds'] is not None]
                center = median(values)
                dispersion.append(dict(case=g['case'], method=g['method'], status=status, phase=phase,
                                       planned_repetitions=g['repetitions'], jobs_with_status=len(selected), observations=len(values),
                                       minimum=min(values, default=None), median=center, maximum=max(values, default=None),
                                       median_absolute_deviation=median(abs(v-center) for v in values)))
    write_csv(output/'timing-dispersion.csv', dispersion, list(dispersion[0]))
    if legacy is not None:
        old = json.loads(legacy.read_text())
        old_by = {(g['case'],g['method']):g for g in old['groups']}
        comparison=[]
        for case in cases:
            for method in ('nuts_cov_diag','nuts_cov_dense','nuts_fisher_diag','nuts_fisher_rank4'):
                new=by.get((case,method));before=old_by.get((case,method))
                if new is None or before is None:
                    continue
                values=dict(case=case,method=method,old_manifest=old['manifest_sha256'],new_manifest=report['manifest_sha256'])
                for prefix,group in [('version2',before),('version3',new)]:
                    for key in ('completed','repetitions','worst_rhat','smallest_bulk_ess','median_maximum_abs_mean_error_reference_sd','median_maximum_abs_sd_ratio_error','median_process_seconds'):
                        values[prefix+'_'+key]=group[key]
                comparison.append(values)
        write_csv(output/'warmup-before-after.csv',comparison,list(comparison[0]))
    print(json.dumps(dict(summary=summary, paired=comparisons), indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--legacy-report',type=Path)
    arguments=parser.parse_args()
    analyze(arguments.run,arguments.output,arguments.legacy_report)
