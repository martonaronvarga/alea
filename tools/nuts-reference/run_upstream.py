"""Verify/fetch the pinned nuts-rs checkout, then execute the locked isolated harness."""
import argparse
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
REV='a762aae513bf7bb5ccb27ffc9a938a923fc837ec'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    checkout=ROOT/'target/nuts-reference-upstream'
    if not checkout.exists():
        checkout.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','init','-q',str(checkout)],check=True)
        subprocess.run(['git','-C',str(checkout),'fetch','-q','--depth=1','https://github.com/pymc-devs/nuts-rs.git',REV],check=True)
        subprocess.run(['git','-C',str(checkout),'checkout','-q','--detach','FETCH_HEAD'],check=True)
    head=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(checkout),'status','--porcelain'],text=True)
    if head!=REV or dirty:raise RuntimeError('reference checkout is not clean and pinned; refusing to overwrite it')
    subprocess.run(['cargo','run','--locked','--bin','alea-nuts-upstream-reference','--manifest-path',str(ROOT/'tools/nuts-reference/upstream/Cargo.toml'),
                    '--',str(args.output.resolve())],check=True)

if __name__=='__main__':main()
