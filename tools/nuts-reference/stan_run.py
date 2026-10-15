"""Run the compiled pinned Stan model, retaining original CSVs and all samples."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import os
import re

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path)
    parser.add_argument('directory',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    timing_path=os.environ.get('ALEA_BENCH_TIMINGS')
    timings=[]
    args.directory.mkdir(exist_ok=False)
    with args.output.open('x',newline='') as out:
        writer=csv.writer(out)
        writer.writerow(('target','seed','draw','q0','q1','divergent','steps','acceptance'))
        for t in range(4):
            data=args.directory/f'data-{t}.json'
            data.write_text(json.dumps(dict(which=t)))
            for seed in range(501,505):
                sign=-1 if seed%2==0 else 1
                init=args.directory/f'init-{seed}.json'
                init.write_text(json.dumps(dict(q=[sign,-sign])))
                output=args.directory/f'{t}-{seed}.csv'
                command=[str(args.model.resolve()),'sample','num_warmup=1024','num_samples=8192',
                         'adapt','engaged=0','algorithm=hmc','engine=nuts','max_depth=8',
                         'metric=unit_e',f'stepsize={0.06 if t==3 else 0.15}',
                         'data',f'file={data}','random',f'seed={seed}',f'init={init}',
                         'output',f'file={output}','sig_figs=17','refresh=0']
                if timing_path:command.insert(4,'save_warmup=1')
                subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
                with output.open() as source:
                    lines=source.readlines()
                    rows=list(csv.DictReader(line for line in lines if not line.startswith('#')))
                if timing_path:
                    warm_steps=sum(int(row['n_leapfrog__']) for row in rows[:1024])
                    rows=rows[1024:]
                    text=''.join(lines)
                    warm=float(re.search(r'([\d.eE+-]+) seconds \(Warm-up\)',text)[1])
                    sample=float(re.search(r'([\d.eE+-]+) seconds \(Sampling\)',text)[1])
                    timings.append((t,seed,warm,sample,warm_steps))
                assert len(rows)==8192
                for draw,row in enumerate(rows):
                    writer.writerow((t,seed,draw,row['q.1'],row['q.2'],row['divergent__'],row['n_leapfrog__'],row['accept_stat__']))
            print(f'Stan completed target {t}',flush=True)
    if timing_path:
        with Path(timing_path).open('x',newline='') as out:
            writer=csv.writer(out)
            writer.writerow(('target','seed','warmup_seconds','sampling_seconds','warmup_steps'))
            writer.writerows(timings)

if __name__=='__main__':main()
