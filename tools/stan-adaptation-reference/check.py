"""Offline triangulation of executed Stan/BlackJAX adaptation fixtures."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def read(path):
 return list(csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith('#')))

def check():
 provenance=json.loads((HERE/'provenance.json').read_text())
 assert provenance['stan']=='2.38.0' and provenance['returncode']==0
 for name,sha in provenance['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
 stan=read(HERE/'schedule.csv');blackjax=read(ROOT/'crates/alea-mcmc/tests/fixtures/blackjax-schedule.csv')
 assert len(stan)==len(blackjax)==1711
 differences=[]
 for a,b in zip(stan,blackjax):
  assert all(a[k]==b[k] for k in ('length','index','slow'))
  if a['end']!=b['end']:differences.append((int(a['length']),int(a['index']),int(a['end']),int(b['end'])))
 assert differences==[(20,17,0,1),(21,18,0,1),(149,134,0,1),(151,99,1,0)],differences
 stan=read(HERE/'trace.csv');blackjax=read(ROOT/'crates/alea-mcmc/tests/fixtures/blackjax-adaptation.csv')
 assert len(stan)==len(blackjax)==400
 maximum=0.
 for a,b in zip(stan,blackjax):
  assert all(a[k]==b[k] for k in ('kind','index'))
  assert all(float(a[k])==float(b[k]) for k in ('x','y','acceptance'))
  for key in ('step','c00','c01','c11'):
   error=abs(float(a[key])-float(b[key]))/(1+abs(float(b[key])))
   assert error<1e-9,(a['kind'],a['index'],key,error)
   maximum=max(maximum,error)
 print('Executed Stan2.38/BlackJAX1.5:1711 schedule rows (4 declared window-end differences),400 recurrence rows; max scaled error',maximum)

if __name__=='__main__':check()
