import json
from pathlib import Path
import tempfile
import unittest

from gaussian_reference import report
from bench import sha


class GaussianReferenceTests(unittest.TestCase):
    def test_replay_checks_raw_scores_hashes_missing_and_duplicate_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = dict(files={}, cases=['linear'], reps=1, evaluations=10,
                            engines=['alea','stan_glm','jax'])
            (out/'manifest.json').write_text(json.dumps(manifest))
            records = []
            for engine in manifest['engines']:
                folder = out/engine
                folder.mkdir()
                raw = folder/'stdout.txt'
                raw.write_text(json.dumps(dict(evaluations=10,evaluation_seconds=1.,
                    logp=-2.,checksum=-20.,gradient=[1.],gradient_checksum=10.)))
                records.append(dict(case='linear',rep=0,engine=engine,folder=engine,
                    returncode=0,files={f'{engine}/stdout.txt':sha(raw)}))
            journal = out/'records.jsonl'
            def write(rows):
                journal.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            write(records[:2])
            self.assertEqual(report(out)['unrecorded_or_unstarted'],1)
            write(records)
            self.assertEqual(report(out)['summary'][0]['qualified'],1)
            raw = out/'jax/stdout.txt'
            raw.write_text(raw.read_text().replace('"gradient": [1.0]','"gradient": [2.0]'))
            with self.assertRaisesRegex(ValueError,'changed reference output'):
                report(out)
            records[2]['files']['jax/stdout.txt'] = sha(raw)
            write(records)
            self.assertEqual(report(out)['summary'][0]['incompatible'],[0])
            write(records+[records[0]])
            with self.assertRaisesRegex(ValueError,'duplicate attempt'):
                report(out)


if __name__ == '__main__':
    unittest.main()
