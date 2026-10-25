"""Offline complete evidence verification with current and archived readers."""
import json,subprocess,sys,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/bench'))
from alea_bench import schema
from alea_bench.manifest import file_hash
from alea_bench.pdb import load
from alea_bench.pdb_bundle import unpack
from alea_bench.pdb_report import build_report

def main():
 evidence=ROOT/'docs/benchmarks/m16/posteriordb-v2'
 with tempfile.TemporaryDirectory() as temp:
  directory=Path(temp);bundle=unpack(evidence,directory/'restored');manifest,rows=load(bundle);report=build_report(bundle)
  assert len(rows)==1560 and len(manifest['corpus']['cases'])==20 and len(manifest['protocol']['methods'])==26,'planned matrix drift'
  assert report==schema.load(bundle/'report.json')==schema.load(evidence/'report.json'),'report regeneration drift'
  assert manifest==schema.load(evidence/'manifest.json'),'manifest drift'
  assert report['purpose']=='descriptive' and 'promotion' not in report,'unexpected decision output'
  assert sum(report['summary']['status_counts'].values())==len(rows),'lost outcome'
  source=directory/'source';source.mkdir()
  with tarfile.open(bundle/'sources.tar.gz','r:gz') as archive:archive.extractall(source,filter='data')
  regenerated=directory/'archived-report.json'
  subprocess.run([sys.executable,str(source/'tools/bench/alea-bench'),'pdb','report',str(bundle),'--output',str(regenerated)],cwd=source,check=True)
  assert schema.load(regenerated)==report,'archived reader report drift'
  analysis=directory/'analysis';analysis.mkdir()
  subprocess.run([sys.executable,str(evidence/'analysis.py'),str(bundle),str(analysis)],check=True,stdout=subprocess.DEVNULL)
  for name in ('findings.md','paired-baseline-costs.json','failures.csv','phase-costs.csv','timing-dispersion.csv'):
   assert file_hash(analysis/name)==file_hash(evidence/name),'additional descriptive analysis drift: '+name
  print(f'M16 descriptive PosteriorDB: {len(rows)} jobs, all raw draws/failures/source seals verified; current and archived reports identical')
if __name__=='__main__':main()
