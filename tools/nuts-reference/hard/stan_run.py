"""Preserve native Stan draws and native CSV-inclusive phase times."""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from models import CASES,SEEDS,WARMUP,DRAWS,DEPTH,data,initial

def main():
    case=int(sys.argv[1]);output=Path(sys.argv[2]);timing=Path(sys.argv[3]);model=Path(sys.argv[4]).resolve()
    raw=output.with_suffix('.raw');raw.mkdir(exist_ok=False)
    dataset=raw/'data.json';dataset.write_text(json.dumps(data(case)))
    _,d,step=CASES[case]
    with output.open('x') as out,timing.open('x') as times:
        writer=csv.writer(out);writer.writerow(['seed','draw','divergent','steps']+[f'q{i}' for i in range(d)])
        tw=csv.writer(times);tw.writerow(('seed','warmup_seconds','sampling_seconds','warmup_steps'))
        for seed in SEEDS:
            init=raw/f'init-{seed}.json';init.write_text(json.dumps(dict(q=initial(d,seed).tolist())))
            result=raw/f'{seed}.csv'
            command=[str(model),'sample',f'num_warmup={WARMUP}',f'num_samples={DRAWS}','save_warmup=1','adapt','engaged=0',
                     'algorithm=hmc','engine=nuts',f'max_depth={DEPTH}','metric=unit_e',f'stepsize={step}',
                     'data',f'file={dataset}','random',f'seed={seed}',f'init={init}','output',f'file={result}','sig_figs=17','refresh=0']
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
            lines=result.read_text().splitlines();rows=list(csv.DictReader(line for line in lines if not line.startswith('#')))
            assert len(rows)==WARMUP+DRAWS
            text='\n'.join(lines)
            wt=float(re.search(r'([\d.eE+-]+) seconds \(Warm-up\)',text)[1]);st=float(re.search(r'([\d.eE+-]+) seconds \(Sampling\)',text)[1])
            tw.writerow((seed,wt,st,sum(int(row['n_leapfrog__']) for row in rows[:WARMUP])))
            for i,row in enumerate(rows[WARMUP:]):writer.writerow([seed,i,row['divergent__'],row['n_leapfrog__']]+[row[f'q.{j+1}'] for j in range(d)])

if __name__=='__main__':main()
