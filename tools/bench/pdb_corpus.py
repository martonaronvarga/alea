"""Vendor a declared, diverse PosteriorDB subset with unchanged model/data bytes."""
import argparse,hashlib,json,math,shutil,statistics,subprocess,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REV='91027f9cdd13b7e76283fec3f2898f59d4b83ec4'
SELECTED=[
 ('Rate_1_data-Rate_1_model','binomial'),('Rate_2_data-Rate_2_model','binomial_contrast'),('Rate_5_data-Rate_5_model','hierarchical_rates'),
 ('arK-arK','autoregression'),('arma-arma11','arma'),('garch-garch11','volatility'),('dogs-dogs_log','count_regression'),
 ('earnings-earn_height','linear_regression'),('earnings-logearn_interaction_z','interaction_regression'),
 ('eight_schools-eight_schools_centered','hierarchical_centered'),('eight_schools-eight_schools_noncentered','hierarchical_noncentered'),
 ('low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse','mixture'),('normal_2-normal_mixture','mixture'),
 ('sblri-blr','logistic'),('wells_data-wells_dist100_model','logistic'),('kidiq-kidscore_momhs','linear_regression'),
 ('seeds_data-seeds_stanified_model','hierarchical_binomial'),('surgical_data-surgical_model','hierarchical_binomial'),
 ('radon_mn-radon_variable_intercept_noncentered','multilevel_regression'),('hmm_example-hmm_example','hidden_markov')]

def read_zip(path):
 with zipfile.ZipFile(path) as z:
  files=[n for n in z.namelist() if n.endswith('.json')]
  if len(files)!=1:raise ValueError('unexpected upstream archive')
  return z.read(files[0])

def generate(checkout):
 assert subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()==REV
 source=checkout/'posterior_database';out=HERE/'corpus/pdb-v2';out.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(checkout/'LICENCE.md',out/'LICENCE.md');cases=[]
 for name,family in SELECTED:
  entry=json.loads((source/'posteriors'/f'{name}.json').read_text());model=entry['model_name'];data=entry['data_name'];directory=out/name;directory.mkdir(exist_ok=True)
  shutil.copyfile(source/'posteriors'/f'{name}.json',directory/'posterior.json')
  for src,dst in [(source/'models/stan'/f'{model}.stan','model.stan'),(source/'models/info'/f'{model}.info.json','model-info.json'),(source/'data/info'/f'{data}.info.json','data-info.json')]:shutil.copyfile(src,directory/dst)
  (directory/'data.json').write_bytes(read_zip(source/'data/data'/f'{data}.json.zip'))
  ref=source/'reference_posteriors/draws/draws'/f'{name}.json.zip';summary={}
  if ref.exists():
   shutil.copyfile(ref,directory/'reference.json.zip');chains=json.loads(read_zip(ref))
   for key in chains[0]:
    values=[v for chain in chains for v in chain[key]]
    if all(isinstance(v,(float,int)) and math.isfinite(v) for v in values):summary[key]={'mean':statistics.mean(values),'sd':statistics.stdev(values),'draws':len(values),'chains':len(chains)}
   (directory/'reference-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
   info=source/'reference_posteriors/draws/info'/f'{name}.info.json'
   if info.exists():shutil.copyfile(info,directory/'reference-info.json')
  files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}
  cases.append({'id':name,'family':family,'directory':directory.relative_to(ROOT).as_posix(),'model':model,'data':data,'files':files,'reference_available':bool(summary),'upstream_url':f'https://github.com/stan-dev/posteriordb/tree/{REV}/posterior_database'})
 corpus={'schema_version':2,'id':'posteriordb-diverse-v2','upstream_revision':REV,'selection':'20 named model/data pairs declared by family before measurement; unchanged upstream data and Stan programs; no claim to cover all PosteriorDB','cases':cases}
 (HERE/'corpus/pdb-v2.json').write_text(json.dumps(corpus,indent=2,sort_keys=True)+'\n');return corpus
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);args=p.parse_args();print(f'vendored {len(generate(args.checkout)["cases"])} posteriors')
