#!/usr/bin/env python3
"""Regenerate nested and clipped continuous Wiener7 reference enclosures."""
import csv,json,math
from pathlib import Path
from wiener7_reference import evaluate

def main():
    path=Path(__file__).with_name('wiener7_nested_balls.csv')
    metadata=[]
    cases=[]
    for lower in [False,True]:
        cases.extend([
            [.03,1.,0.,.5,2.,.5,.1,.005,lower],
            [.03,1.,0.,.5,2.,.5,.1,.05,lower],
            [.03,1.,0.,.5,2.,0.,0.,.03,lower],
            [.2,1.2,.1,.4,-1.,.5,0.,.15,lower],
        ])
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(['rt','alpha','tau','beta','delta','sd','sw','st','lower','precision','terms']+[f'{v}_{s}' for v in ['log','alpha','tau','beta','delta','sd','sw','st'] for s in ['mid','radius']])
        for index,params in enumerate(cases):
            values,info=evaluate(params,goal='1e-15',max_precision=384)
            row=params[:8]+[int(params[8]),info['precision'],info['terms']]
            for x in values:
                mid=float(x.mid());radius=math.nextafter(float(abs(x-mid).upper()),math.inf);row.extend([mid,radius])
            writer.writerow(row);stream.flush();metadata.append(dict(params=params,**info));print(index,info,flush=True)
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
if __name__=='__main__':main()
