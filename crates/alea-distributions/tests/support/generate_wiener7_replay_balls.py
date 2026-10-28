#!/usr/bin/env python3
"""Regenerate captured Wiener7 numerical-proposal ball references.

The high drift-variance case uses exact time integration. The other cases
retain independent continuous time quadrature. Requires python-flint 0.9.0.
"""
import csv,math
from pathlib import Path
from flint import arb
from wiener7_reference import evaluate
from wiener5_cdf_reference import evaluate as evaluate_cdf

CASES = [[0.469, 2.4749736829675593, 0.16668455332354118, 0.4689899247129888, 0.6389540258904112, 1.0066854590943801, 0.7799458999413749, 0.34291442859487303, False], [0.675, 1.3379729514065335, 0.4604482709501249, 0.48721932677946567, 0.6890406129269979, 0.017193277147217887, 0.7909296417576759, 0.3746925547442049, True], [0.675, 1.5779382816950356, 0.3892536621148969, 0.3415871894921585, 0.5005374649192191, 349.4261431791342, 0.6348044548085039, 0.3751468294604952, True], [0.675, 1.184370251981865, 0.3238302317741533, 0.19842378532536026, 2.014406521010839, 0.9636510388824948, 0.2652004524018589, 0.3619964232523812, True], [0.4184013856731187, 1.5178881986936503, 0.2485551790941962, 0.6164806090045881, 1.2061561742322793, 1.2615614023185378, 0.4513465885975576, 0.1704964522307809, False], [0.3807531142811618, 1.4218349687873668, 0.2630065224865575, 0.5287900854310247, 0.6889164658022419, 0.13240361346377336, 0.6492779888742992, 0.11596088802520918, False], [0.36303379407076364, 1.4166544003274433, 0.2877173417664885, 0.5443025249813934, 0.6323720847103405, 0.25203635546122377, 0.6581086385541228, 0.08915750638575942, False], [0.41124039843703153, 1.346855690424418, 0.2411090185689389, 0.5340249572171718, 0.8143655551175284, 0.648245807686055, 0.6034460819045603, 0.2018628898840445, False], [2e-06, 0.7, 1e-06, 0.25, 30.0, 0.01, 1e-08, 0.2, False]]

def main():
    path=Path(__file__).with_name('wiener7_replay_balls.csv')
    with path.open('w',newline='') as stream:
        out=csv.writer(stream,lineterminator='\n')
        out.writerow(['rt','a','tau','beta','v','sd','sw','st','lower','precision','terms']+[f'{v}_{s}' for v in ['log','a','tau','beta','v','sd','sw','st'] for s in ['mid','radius']])
        for index,params in enumerate(CASES):
            values,info=(evaluate_cdf if index==2 else evaluate)(params,goal='1e-11',max_precision=192)
            row=params[:8]+[int(params[8]),info['precision'],info['terms']]
            for x in values:
                m=float(x.mid());row.extend([m,math.nextafter(float(abs(x-arb(m)).upper()),math.inf)])
            out.writerow(row);stream.flush();print(index,info,flush=True)
    print(path)

if __name__=='__main__':main()
