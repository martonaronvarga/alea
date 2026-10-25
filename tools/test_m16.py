"""Validate checked M16 evidence independently of binaries, engines or wall time."""
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/bench'))
from alea_bench import schema
from alea_bench.report import build_report,load_bundle


def main():
    evidence=ROOT/'docs/benchmarks/m16'
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        with tarfile.open(evidence/'reference-v1.tar.gz','r:gz') as archive:
            archive.extractall(directory,filter='data')
        bundle=directory/'reference-v1'
        manifest,records=load_bundle(bundle)
        report=build_report(bundle)
        assert report==schema.load(bundle/'report.json')==schema.load(evidence/'report.json'),'report regeneration drift'
        assert manifest==schema.load(evidence/'manifest.json'),'manifest drift'
        initial=schema.manifest(schema.load(evidence/'attempt-01-manifest.json'))
        assert initial['protocol']==manifest['protocol'] and initial['corpus']==manifest['corpus'],'scientific policy/data changed after the first attempt'
        assert len(records)==252 and set(r['engine'] for r in records)==set(schema.ENGINES),'planned workload drift'
        assert len(report['calibration'])==3 and all(c['datasets']==32 for c in report['calibration']),'calibration workload drift'
        assert not report['promotion']['ledger_changed'],'experiment modified a promotion ledger'
        # A second reader uses the implementation shipped in the measured source
        # archive, without local Rust binaries or the current library on sys.path.
        source=directory/'source';source.mkdir()
        with tarfile.open(bundle/'sources.tar.gz','r:gz') as archive:
            archive.extractall(source,filter='data')
        regenerated=directory/'independent-report.json'
        subprocess.run([sys.executable,str(source/'tools/bench/alea-bench'),'report',str(bundle),'--output',str(regenerated)],check=True,cwd=source)
        assert schema.load(regenerated)==report,'archived implementation cannot regenerate its own report'
        print(f'M16: {len(records)} source-bound planned jobs and full report reproduced; timing is informational')


if __name__=='__main__':main()
