"""Build pinned engines before measured jobs; preserve build logs and durations."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from .manifest import ROOT, file_hash, write_new, source_paths

REV='a762aae513bf7bb5ccb27ffc9a938a923fc837ec'


def prepare(output,engines,root=ROOT):
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    rust_sources={p.relative_to(root).as_posix():file_hash(p) for p in source_paths(root) if p.suffix in (".rs",".toml",".lock",".stan",".c",".h",".hpp",".nix")}
    def build(name,command):
        start=time.perf_counter()
        with (output/f'{name}.log').open('x') as log:
            p=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        records.append({'name':name,'command':command,'seconds':time.perf_counter()-start,'returncode':p.returncode,'log_sha256':file_hash(output/f'{name}.log')})
        write_new(output/f'{name}.build.json',records[-1])
        if p.returncode:raise RuntimeError(f'{name} preparation failed; see {output/name}.log')
    build('alea-fetch',['nix','develop','.#stable','--command','cargo','fetch','--locked','--manifest-path','crates/Cargo.toml'])
    build('alea',['nix','develop','.#stable','--command','cargo','build','--release','--locked','--offline','--manifest-path','crates/Cargo.toml','-p','alea-cli','--bin','alea-bench'])
    binaries={'alea':str(root/'crates/target/release/alea-bench')}
    if 'nuts-rs' in engines:
        checkout=root/'target/nuts-reference-upstream'
        if not checkout.exists():
            checkout.parent.mkdir(exist_ok=True)
            subprocess.run(['git','init','-q',str(checkout)],check=True)
            subprocess.run(['git','-C',str(checkout),'fetch','-q','--depth=1','https://github.com/pymc-devs/nuts-rs.git',REV],check=True)
            subprocess.run(['git','-C',str(checkout),'checkout','-q','--detach','FETCH_HEAD'],check=True)
        head=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(checkout),'status','--porcelain'],text=True)
        if head!=REV or dirty:raise RuntimeError('upstream checkout must be clean and pinned')
        build('nuts-rs-fetch',['nix','develop','.#stable','--command','cargo','fetch','--locked','--manifest-path','tools/nuts-reference/upstream/Cargo.toml'])
        build('nuts-rs',['nix','develop','.#stable','--command','cargo','build','--release','--locked','--offline','--manifest-path','tools/nuts-reference/upstream/Cargo.toml','--bin','m16'])
        binaries['nuts-rs']=str(root/'tools/nuts-reference/upstream/target/release/m16')
    if 'blackjax' in engines:
        start=time.perf_counter();environment=subprocess.check_output(['nix-build','tools/blackjax-reference/default.nix','--no-out-link'],cwd=root,text=True).strip().splitlines()[-1]
        binaries['blackjax']=environment+'/bin/python3'
        records.append({'name':'blackjax','seconds':time.perf_counter()-start,'pin':'tools/blackjax-reference/default.nix'})
    if 'stan' in engines:
        start=time.perf_counter();environment=subprocess.check_output(['nix','build','.#cmdStan','--no-link','--print-out-paths'],cwd=root,text=True).strip().splitlines()[-1]
        environment_seconds=time.perf_counter()-start
        stan_dir=output/'stan';stan_dir.mkdir();source=stan_dir/'targets.stan';shutil.copyfile(root/'tools/bench/targets.stan',source)
        build('stan',['nix','develop','.#stable','--command','make','-C',environment+'/opt/cmdstan',str(source.with_suffix('').resolve()),'PRECOMPILED_MODEL_HEADER='])
        binaries['stan']=str(source.with_suffix('').resolve())
        records[-1]['environment_seconds']=environment_seconds
    if any(file_hash(root/name)!=sha for name,sha in rust_sources.items()):raise RuntimeError('build sources changed during preparation; use a new build directory')
    config={'schema_version':1,'rust_sources':rust_sources,'binaries':binaries,'binary_sha256':{k:file_hash(Path(v)) for k,v in binaries.items()},'records':records,
            'build_environment':{k:os.environ.get(k) for k in ('RUSTFLAGS','RUSTDOCFLAGS','CARGO_BUILD_TARGET','CARGO_PROFILE_RELEASE_LTO','CARGO_PROFILE_RELEASE_OPT_LEVEL','CARGO_PROFILE_RELEASE_DEBUG','CARGO_PROFILE_RELEASE_CODEGEN_UNITS','CC','CXX')},
            'rustc':subprocess.check_output(['nix','develop','.#stable','--command','rustc','--version'],cwd=root,text=True).strip()}
    write_new(output/'engines.json',config)
    return config
