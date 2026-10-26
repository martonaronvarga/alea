#!/usr/bin/env python3
"""Prespecified Wiener7 performance-grid regression enclosures, python-flint 0.9."""
import csv,math
from pathlib import Path
from flint import arb
from wiener7_reference import evaluate


def cases():
    groups=[('no_variability',.5,.5,0.,0.),('start_tiny',.05,.5,1e-8,0.),
            ('start_image',.05,.5,.1,0.),('start_wide',.01,.5,.4,0.),
            ('start_spectral',.5,0.,.4,0.),('time_tiny',.5,0.,0.,1e-6),
            ('time_only',.5,0.,0.,.1),('time_clipped',.03,0.,0.,.1),
            ('both_broad',.05,.5,.1,.005),('both_localized',.01,.5,.4,.001),
            ('both_clipped',.03,.5,.1,.05),('both_long',2.,.5,.1,.1)]
    for group,t,sd,sw,st in groups:
        for drift in [-2.,0.,2.]:
            for lower in [False,True]:
                yield group,[t,1.,0.,.5,drift,sd,sw,st,lower]
    for lower in [False,True]:
        for group,st in [('extreme_tail',0.),('extreme_joint',1e-8)]:
            yield group,[1e-6,.1,0.,.0001,-10.,.1,1e-8,st,lower]


def save(rows):
    path=Path(__file__).with_name('wiener7_grid_balls.csv')
    with path.open('w',newline='') as stream:
        out=csv.writer(stream,lineterminator='\n')
        out.writerow(['group','rt','alpha','tau','beta','delta','sd','sw','st','lower']+
                     [f'{v}_{s}' for v in ['log','alpha','tau','beta','delta','sd','sw','st'] for s in ['mid','radius']])
        for group,params,values in rows:
            row=[group]+params[:8]+[int(params[8])]
            for x in values:
                mid=float(x.mid());radius=math.nextafter(float(abs(x-arb(mid)).upper()),math.inf)
                row.extend([mid,radius])
            out.writerow(row);stream.flush()
    return path


def main():
    cache={};rows=[]
    for group,params in cases():
        # Reflection is exact at beta=1/2: swap boundaries, negate drift,
        # and negate beta/drift scores. Other parameters and scores agree.
        reflected=params[3]==.5 and params[8]
        source=params.copy()
        if reflected:source[4]=-source[4];source[8]=False
        key=tuple(source)
        if key not in cache:
            cache[key]=evaluate(source,goal='1e-15',max_precision=384)[0]
        values=cache[key].copy()
        if reflected:values[3]=-values[3];values[4]=-values[4]
        rows.append((group,params,values))
        print(group,params[4],params[8],flush=True)
    print(save(rows))
if __name__=='__main__':main()
