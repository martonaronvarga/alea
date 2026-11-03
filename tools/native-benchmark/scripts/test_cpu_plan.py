import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cpu_plan as cpu


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


class CampaignTests(unittest.TestCase):
    def test_construction_scope_is_explicit_and_cannot_silently_change(self):
        self.assertIsNone(cpu.construction_scope('historical receipt'))
        marker = 'ALEA_CPU_PLAN_BUILD_SCOPE shared-prepared-data-v2\n'
        self.assertEqual(cpu.construction_scope(marker), 'shared-prepared-data-v2')
        for text in (marker + marker, marker.replace('shared-prepared-data-v2', 'unknown')):
            with self.assertRaises(ValueError):
                cpu.construction_scope(text)
        self.run_fixture(0)
        self.run_fixture(1)
        before = cpu.rebuild(self.root)
        after = copy.deepcopy(before)
        for row in after['records']:
            row['construction_scope'] = 'shared-prepared-data-v2'
        with patch.object(cpu, 'rebuild', side_effect=[before, after]), self.assertRaisesRegex(ValueError, 'construction scope'):
            cpu.compare_campaigns(self.root, self.root)
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'benchmark').write_bytes(b'fake-test-binary')
        (self.root / 'sources.tar.gz').write_bytes(b'fake-test-source-seal')
        self.manifest = dict(schema_version=cpu.VERSION, suite='smoke', reps=2,
                             binary_sha256=cpu.sha(self.root / 'benchmark'),
                             source_archive_sha256=cpu.sha(self.root / 'sources.tar.gz'),
                             criterion=dict(samples=10))
        write(self.root / 'manifest.json', self.manifest)

    def run_fixture(self, rep=0):
        folder = self.root / f'rep-{rep:02}'
        folder.mkdir()
        lines = []
        for family, n, p in cpu.cases('smoke'):
            lines.append(f'ALEA_CPU_PLAN_BEGIN {family} {n} {p}')
            for mode in cpu.MODES:
                score = [1.] * (p + (family == 'normal'))
                lines.append(f'ALEA_CPU_PLAN_VALUE {family} {n} {p} {mode} -12.0 {json.dumps(score)}')
                for phase in cpu.PHASES:
                    directory = folder / 'raw' / f'regression_plan_{phase}_{family}_{n}_{p}' / mode / 'new'
                    full = f'regression_plan/{phase}/{family}/{n}/{p}/{mode}'
                    ns = 1000. if mode == 'Scalar' else 2000.
                    write(directory / 'benchmark.json', dict(full_id=full, group_id=full.rsplit('/', 1)[0],
                          function_id=mode, throughput={'Elements': 1} if phase == 'evaluate' else None))
                    write(directory / 'estimates.json', dict(mean=dict(point_estimate=ns, confidence_interval=dict(
                          confidence_level=0.95, lower_bound=ns * .9, upper_bound=ns * 1.1))))
                    write(directory / 'sample.json', dict(iters=[1.] * 10, times=[ns] * 10))
        (folder / 'stderr.txt').write_text('\n'.join(lines) + '\n')
        (folder / 'stdout.txt').write_text('Criterion fixture\n')
        write(folder / 'run.json', dict(schema_version=cpu.VERSION, suite='smoke', rep=rep,
              sealed=True, inputs_verified=True, returncode=0,
              binary_sha256=self.manifest['binary_sha256'], source_archive_sha256=self.manifest['source_archive_sha256'],
              files=cpu.run_files(folder)))
        return folder

    def reseal(self, folder):
        run = cpu.read_metrics((folder / 'run.json').read_text())
        run['files'] = cpu.run_files(folder)
        write(folder / 'run.json', run)

    def test_grids_are_disjoint_and_preserve_all_planned_families(self):
        self.assertEqual(len(cpu.cases('calibration')), 45)
        self.assertEqual(len(cpu.cases('validation')), 36)
        self.assertFalse(set(cpu.cases('calibration')) & set(cpu.cases('validation')))

    def test_source_guard_rejects_added_inputs_removed_inputs_and_executable_mode_changes(self):
        path = self.root / 'fixture.rs'
        path.write_text('source')
        entry = dict(path='fixture.rs', sha256=cpu.sha(path), mode=0o644)
        with patch.object(cpu, 'REPO', self.root), patch.object(cpu.source_archive, 'source_files', return_value=[path]):
            self.assertTrue(cpu.unchanged([entry]))
            self.assertFalse(cpu.unchanged([]))
            path.chmod(0o755)
            self.assertFalse(cpu.unchanged([entry]))
        with patch.object(cpu, 'REPO', self.root), patch.object(cpu.source_archive, 'source_files', return_value=[]):
            self.assertFalse(cpu.unchanged([entry]))

    def test_complete_pairs_report_throughput_regret_and_amortization(self):
        self.run_fixture(0)
        self.run_fixture(1)
        for item in cpu.rebuild(self.root)['summary']:
            self.assertEqual((item['successful'], item['failed'], item['unstarted']), (2, 0, 0))
            self.assertEqual(item['phases']['evaluate']['auto_regret']['median'], 1.)
            self.assertEqual(item['evaluations_per_second']['Auto']['median'], 500000.)
            self.assertEqual(item['amortized']['1000']['auto_over_scalar']['median'], 2.)

    def test_partial_case_survives_failed_child_without_admitting_missing_candidate(self):
        folder = self.run_fixture()
        lines = (folder / 'stderr.txt').read_text().splitlines()
        (folder / 'stderr.txt').write_text('\n'.join(line for line in lines if 'VALUE logit' not in line))
        self.reseal(folder)
        run = cpu.read_metrics((folder / 'run.json').read_text())
        run['returncode'] = 101
        write(folder / 'run.json', run)
        rows = {r['family']: r for r in cpu.rebuild(self.root)['summary']}
        self.assertEqual(rows['normal']['successful'], 1)
        self.assertEqual(rows['logit']['failed'], 1)
        self.assertEqual(rows['normal']['unstarted'], 1)
        self.assertIsNone(rows['logit']['phases']['evaluate']['auto_regret'])

    def test_changed_or_unsealed_run_has_no_eligible_timings(self):
        folder = self.run_fixture()
        (folder / 'stdout.txt').write_text('tampered')
        self.assertTrue(all(not r['success'] for r in cpu.rebuild(self.root)['records']))
        self.reseal(folder)
        run = cpu.read_metrics((folder / 'run.json').read_text())
        for mutation in (dict(sealed=False), dict(inputs_verified=False), dict(binary_sha256='other'),
                         dict(source_archive_sha256='other'), dict(schema_version=True), dict(rep=False)):
            write(folder / 'run.json', {**run, **mutation})
            self.assertTrue(all(not r['success'] for r in cpu.rebuild(self.root)['records']))

    def test_case_never_started_in_failed_child_stays_unstarted(self):
        folder = self.run_fixture()
        lines = (folder / 'stderr.txt').read_text().splitlines()
        (folder / 'stderr.txt').write_text('\n'.join(v for v in lines if 'poisson' not in v))
        self.reseal(folder)
        result = cpu.rebuild(self.root)['summary'][2]
        self.assertEqual((result['successful'], result['failed'], result['unstarted']), (0, 0, 2))

    def test_binary_or_source_corruption_rejects_campaign(self):
        for name in ('benchmark', 'sources.tar.gz'):
            original = (self.root / name).read_bytes()
            (self.root / name).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                cpu.rebuild(self.root)
            (self.root / name).write_bytes(original)

    def test_missing_sample_file_does_not_reuse_old_estimate_or_cached_summary(self):
        folder = self.run_fixture()
        path = folder / 'raw/regression_plan_evaluate_normal_1000_4/Auto/new/sample.json'
        path.unlink()
        self.reseal(folder)
        write(self.root / 'summary.json', {'success': True, 'all_fast': True})
        normal = cpu.rebuild(self.root)['summary'][0]
        self.assertEqual(normal['successful'], 0)
        self.assertIsNone(normal['phases']['evaluate']['ns']['Auto'])

    def test_recompute_numerical_admission_rejects_disagreement_and_wrong_dimension(self):
        folder = self.run_fixture()
        original = (folder / 'stderr.txt').read_text()
        for altered in (original.replace('Auto -12.0', 'Auto -13.0'),
                        original.replace('[1.0, 1.0, 1.0, 1.0, 1.0]', '[1.0]'),
                        original + original.splitlines()[0] + '\n',
                        original.replace('-12.0', 'NaN'),
                        original.replace('1.0', 'true')):
            (folder / 'stderr.txt').write_text(altered)
            self.reseal(folder)
            self.assertTrue(all(not r['success'] for r in cpu.rebuild(self.root)['records']))

    def test_criterion_identity_and_work_validation_are_not_cached_flags(self):
        folder = self.run_fixture()
        base = folder / 'raw/regression_plan_evaluate_normal_1000_4/Auto/new'
        for name, mutate in (
            ('benchmark.json', lambda v: v.update(full_id='wrong')),
            ('benchmark.json', lambda v: v.update(throughput={'Elements': 1000})),
            ('sample.json', lambda v: v.update(iters=[1.])),
            ('sample.json', lambda v: v.update(times=[0.] * 10)),
            ('sample.json', lambda v: v.update(iters=[0.5] * 10)),
            ('estimates.json', lambda v: v['mean'].update(point_estimate=-1.)),
            ('estimates.json', lambda v: v['mean']['confidence_interval'].update(lower_bound=3000.)),
        ):
            path = base / name
            original = cpu.read_metrics(path.read_text())
            changed = copy.deepcopy(original)
            mutate(changed)
            write(path, changed)
            self.reseal(folder)
            self.assertFalse(cpu.rebuild(self.root)['records'][0]['success'])
            write(path, original)
            self.reseal(folder)

    def test_duplicate_repetitions_and_nonfinite_json_fail_explicitly(self):
        folder = self.run_fixture()
        rows = cpu.rebuild(self.root)['records']
        with self.assertRaises(ValueError):
            cpu.summarize(rows + rows, 'smoke', 2)
        (folder / 'run.json').write_text('{"schema_version":1,"schema_version":1}')
        self.assertTrue(all(not r['success'] for r in cpu.rebuild(self.root)['records']))

    def test_comparison_rejects_setting_drift_and_bit_changes_including_signed_zero(self):
        self.run_fixture(0)
        self.run_fixture(1)
        for item in cpu.compare_campaigns(self.root, self.root)['summary']:
            self.assertEqual(item['qualified'], 2)
            self.assertEqual(item['phases']['evaluate']['Auto']['paired_after_over_before']['median'], 1.)
        before = cpu.rebuild(self.root)
        after = copy.deepcopy(before)
        before['records'][0]['values']['Auto'][1] = 0.
        after['records'][0]['values']['Auto'][1] = -0.
        with patch.object(cpu, 'rebuild', side_effect=[before, after]):
            self.assertEqual(cpu.compare_campaigns(self.root, self.root)['summary'][0]['withheld'], 1)
        manifest = copy.deepcopy(self.manifest)
        before_manifest, after_manifest = dict(manifest, cpu=2), dict(manifest, cpu=3)
        with patch.object(cpu, 'read_metrics', side_effect=[before_manifest, after_manifest]), patch.object(cpu, 'rebuild', return_value=before):
            with self.assertRaisesRegex(ValueError, 'cpu'):
                cpu.compare_campaigns(self.root, self.root)


if __name__ == '__main__':
    unittest.main()
