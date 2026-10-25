"""Seal numerical policy and actual source bytes before any measurement."""
import hashlib
import json
import platform
import os
import subprocess
from pathlib import Path
from . import schema

ROOT=Path(__file__).resolve().parents[3]


def file_hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path,value):
    with path.open('x') as out:json.dump(value,out,indent=2,sort_keys=True,allow_nan=False);out.write('\n')


def source_paths(root):
    paths=[]
    for base in ('crates','tools/bench'):
        for directory,dirs,files in os.walk(root/base):
            dirs[:]=[d for d in dirs if d not in ('target','__pycache__','.git')]
            for name in files:
                path=Path(directory)/name
                if path.suffix in ('.rs','.toml','.lock','.py','.stan','.json','.nix','.cpp','.hpp','.txt','.csv'):paths.append(path)
    # build.rs compiles hcubature only. The unused 40 MiB pcubature table is
    # deliberately excluded; retain every native input and its upstream license.
    paths += [root/'crates/alea-ffi'/p for p in ('vendor/cubature/hcubature.c','vendor/cubature/cubature.h','vendor/cubature/converged.h','vendor/cubature/vwrapper.h','vendor/cubature/COPYING.md','tests/cubature_cleanup.c')]
    paths += list((root/"tools/bench/corpus/vendor").glob("*"))
    paths += [root/p for p in ("tools/bench/alea-bench","tools/nuts-reference/upstream/src/bin/m16.rs",'.gitmodules','flake.lock','flake.nix','treefmt.nix','tools/blackjax-reference/default.nix','tools/nuts-reference/upstream/Cargo.lock','tools/nuts-reference/upstream/Cargo.toml','tools/nuts-reference/upstream/src/bin/m16.rs','tools/ci/maturity-ledger.json','tools/ci/exactness-ledger.json')]
    return sorted(set(paths))


def seal(protocol_path,output,root=ROOT):
    p=schema.protocol(schema.load(protocol_path));corpus=schema.corpus(schema.load(root/p['corpus']))
    sources={path.relative_to(root).as_posix():file_hash(path) for path in source_paths(root)}
    sources[p['corpus']]=file_hash(root/p['corpus'])
    sources[protocol_path.resolve().relative_to(root).as_posix()]=file_hash(protocol_path)
    git={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'worktree_content_is_authoritative':True}
    ledgers={name:{'sha256':file_hash(root/f'tools/ci/{name}-ledger.json'),'content':schema.load(root/f'tools/ci/{name}-ledger.json') if name=='exactness' else {'profiles':schema.load(root/f'tools/ci/{name}-ledger.json')['profiles']}} for name in ('maturity','exactness')}
    toolchains={'rust':'flake.lock stable shell','stan':'2.38.0 (flake.lock)','blackjax':'1.5','jax':'0.9.2','jaxlib':'0.9.2','nuts-rs':'a762aae513bf7bb5ccb27ffc9a938a923fc837ec','python_controller':platform.python_version()}
    m=dict(schema_version=1,protocol=p,corpus=corpus,sources=sources,git=git,toolchains=toolchains,ledgers=ledgers,protocol_sha256=schema.digest(p),corpus_sha256=schema.digest(corpus))
    m['manifest_sha256']=schema.digest(m);schema.manifest(m);write_new(output,m);return m


def verify_sources(manifest,root=ROOT):
    for name,expected in manifest['sources'].items():
        path=root/schema.relative(name)
        if not path.is_file() or file_hash(path)!=expected:raise schema.Invalid(f'source changed after seal: {name}; create a new protocol/manifest, never rewrite a measured seal')


def task_text(case,policy):
    values=[1,case['kind'],case['dimension'],case['rows'],policy['warmup'],policy['draws'],policy['depth'],case['step'],policy['chains'],len(case['data']),*policy['seeds'],*case['initial'],*case['data']]
    return ' '.join(str(v) for v in values)+'\n'
