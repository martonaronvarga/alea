"""Native-backend preparation, conformance and descriptive PosteriorDB campaigns."""
import argparse
import ctypes
import json
import os
import subprocess
import shutil
import time
from pathlib import Path

from . import pdb, schema
from .manifest import ROOT, file_hash, source_paths, write_new
from .native_ports import generate
from .runner import environment


def dependency_sources():
    result={}
    for base in ('target/nuts-reference-upstream','target/m16b-source/bridgestan/rust'):
        for directory,dirs,files in os.walk(ROOT/base):
            dirs[:]=[d for d in dirs if d not in ('.git','target')]
            for name in files:
                path=Path(directory)/name
                if path.suffix in ('.rs','.toml','.lock','.md','.h','.c','.txt') or name.startswith(('LICENSE','COPYING','NOTICE')):
                    result[path.relative_to(ROOT).as_posix()]=file_hash(path)
    if not result: raise ValueError('native worker source dependencies unavailable')
    return result


def build(output, models):
    """Build one source-bound native worker; reuse hash-verified Stan executables."""
    output = output.resolve(); output.mkdir(parents=True, exist_ok=False)
    ports = output/'ports'; inventory = generate(ports)
    # Evaluate shells from a tiny copy of the identical flake. A source archive
    # without Git must not copy gigabytes of growing Cargo targets into Nix.
    toolchain=output/'toolchain';toolchain.mkdir();(toolchain/'crates').mkdir()
    for name in ('flake.nix','flake.lock','treefmt.nix'):shutil.copyfile(ROOT/name,toolchain/name)
    shutil.copyfile(ROOT/'crates/Cargo.toml',toolchain/'crates/Cargo.toml')
    flake='path:'+str(toolchain)
    libclang = subprocess.check_output(['nix', 'build', '--impure', '--no-link', '--print-out-paths', '--expr',
        'let lock=builtins.fromJSON(builtins.readFile '+json.dumps(str(ROOT/'flake.lock'))+'); source=builtins.fetchTree lock.nodes.nixpkgs.locked; p=import source {system=builtins.currentSystem;}; in p.llvmPackages.libclang.lib'], text=True).strip().splitlines()[-1]
    env = dict(environment(ROOT), ALEA_NATIVE_PORTS=str(ports), LIBCLANG_PATH=libclang+'/lib',
               CARGO_TARGET_DIR=str(ROOT/'target/native-pdb-enzyme'))
    command = ['nix', 'develop', flake+'#autodiff', '--command', 'cargo', 'build', '--release', '--locked',
               '--features', 'native-enzyme', '--manifest-path', 'tools/bench/pdb-worker/Cargo.toml']
    inputs = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_paths(ROOT)}
    dependencies=dependency_sources();inputs.update(dependencies)
    start = time.perf_counter_ns()
    with (output/'build.log').open('x') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    write_new(output/'build.json', dict(command=command, returncode=result.returncode,
        seconds=(time.perf_counter_ns()-start)/1e9, compiled_source_inputs=inputs,
        scope='one batch worker build; incremental cache may be reused; not a per-model cold compilation comparison'))
    if result.returncode: raise RuntimeError('native worker build failed; see build.log')
    if any(file_hash(ROOT/p)!=sha for p, sha in inputs.items()): raise ValueError('sources changed during native compilation')
    pdb.WORKER = ROOT/'target/native-pdb-enzyme/release/alea-pdb-worker'
    pdb.JAX = Path(subprocess.check_output(['nix-build', 'tools/blackjax-reference/default.nix', '--no-out-link'], cwd=ROOT, text=True).strip().splitlines()[-1])/'bin/python3'
    stan = Path(subprocess.check_output(['nix','build',flake+'#cmdStan','--no-link','--print-out-paths'],cwd=ROOT,text=True).strip().splitlines()[-1])/'opt/cmdstan'
    tbb = stan/'stan/lib/stan_math/lib/tbb/libtbb.so.2'
    runtime_environment = {'LD_LIBRARY_PATH':str(tbb.parent)}
    # Reuse only byte-verified source/model/binary records, never old worker seals.
    corpus = schema.load(ROOT/'tools/bench/corpus/pdb-v3.json')
    cases = {c['id']: c for c in corpus['cases']}
    records = schema.load(models/'models.json')
    for row in records['records']:
        case = cases[row['case']]
        for item in row['builds'].values():
            if item['returncode'] == 0 and file_hash(Path(item['binary'])) != item['binary_sha256']: raise ValueError('reused Stan binary drift')
            if file_hash(Path(item['binary']).parent/'model.stan') != case['files']['model.stan']: raise ValueError('reused Stan source drift')
    prepared = pdb.prepare_metadata(models, output/'metadata', dict(environment(ROOT), **runtime_environment))
    prepared.update(schema_version=3, native_directory=str(output),
                    third_ad_source_sha256=file_hash(ROOT/'tools/bench/third_ad.cpp'))
    prepared['native'] = dict(schema_version=1, compiler=subprocess.check_output(['nix', 'develop', flake+'#autodiff', '--command', 'rustc', '-Vv'], cwd=ROOT, text=True).strip(),
        rust_backend='enzyme', execution='serial', conformance={}, inventory=inventory,
        model_frontend='explicit native Rust primal through Alea ModelBuilder source export',
        nuts_rs_frontend='same generated Enzyme target through CpuLogpFunc; no separate model compiler',
        runtime_environment=runtime_environment,tbb_library=str(tbb),tbb_sha256=file_hash(tbb),
        dependency_sources=dependencies,
        compiled_ports={p.name:file_hash(p) for p in ports.iterdir() if p.is_file()})
    prepared['compiled_source_inputs'] = inputs
    prepared['build_records']['native_worker'] = schema.load(output/'build.json')
    write_new(output/'built.json', prepared)
    return prepared


def conform(prepared_path, output):
    from .native_conformance import validate_case
    output = output.resolve(); output.mkdir(parents=True, exist_ok=False)
    prepared = schema.load(prepared_path)
    ctypes.CDLL(prepared['native']['tbb_library'], mode=ctypes.RTLD_GLOBAL)
    cases = schema.load(ROOT/'tools/bench/corpus/pdb-v3.json')['cases']
    for case in cases:
        try:
            prepared['native']['conformance'][case['id']] = validate_case(case, prepared['models'][case['id']],
                prepared['worker'], prepared['jax_python'], output/case['id'], prepared['native']['runtime_environment'])
        except Exception as error:
            prepared['native']['conformance'][case['id']] = dict(status='failed', reason=f'{type(error).__name__}: {error}')
        print(case['id'], prepared['native']['conformance'][case['id']], flush=True)
    prepared['native']['conformance_directory'] = str(output)
    prepared['native_conformance_directory'] = str(output)
    prepared['native']['conformance_files'] = {p.relative_to(output).as_posix():file_hash(p) for p in output.rglob('*') if p.is_file()}
    write_new(output/'prepared.json', prepared)
    return output/'prepared.json'


def execute(protocol_path, prepared_path, output, backend='enzyme', execution='serial'):
    output.mkdir(parents=True, exist_ok=False)
    prepared = schema.load(prepared_path)
    for name, digest in prepared['native']['compiled_ports'].items():
        if file_hash(Path(prepared['native_directory'])/'ports'/name) != digest: raise ValueError('compiled native primal drift')
    for name, digest in prepared['native']['conformance_files'].items():
        if file_hash(Path(prepared['native_conformance_directory'])/name) != digest: raise ValueError('native conformance artifact drift')
    prepared['native'].update(rust_backend=backend, execution=execution)
    write_new(output/'prepared.json', prepared)
    pdb.seal(protocol_path, output/'prepared.json', output/'manifest.json')
    pdb.run(output/'manifest.json', output/'run')
    from .pdb_report import write_report
    report = write_report(output/'run')
    write_new(output/'report.json', report)
    from .native_report import write
    write(output)
    return report


def campaign(output, models):
    """Build, validate, pilot, then execute immutable broad comparison protocols."""
    output=output.resolve(); output.mkdir(parents=True,exist_ok=False)
    build(output/'preparation',models)
    prepared=conform(output/'preparation/built.json',output/'conformance')
    # Conformance failures are recorded model-by-model. An invalid pilot model
    # stops automatic expansion; it never becomes a silently different backend.
    pilot=ROOT/'tools/bench/protocols/posteriordb-native-pilot-v3.json'
    config=schema.load(prepared)
    invalid=[c for c in schema.load(pilot)['cases'] if config['native']['conformance'][c]['status']!='validated']
    if invalid: raise RuntimeError('pilot native conformance failed: '+', '.join(invalid))
    report=execute(pilot,prepared,output/'pilot')
    pipeline_failures=[r for r in report['rows'] if r['status']=='failed' and (r['failure'] or {}).get('kind') in ('Invalid','FileNotFoundError','KeyError')]
    if pipeline_failures: raise RuntimeError('pilot pipeline failure; broad execution stopped with artifacts preserved')
    # A sampler failure is an observation, not a statistical exclusion gate.
    for name,backend,execution in [('broad','enzyme','serial'),('shared-stan','bridgestan','serial'),
                                   ('parallel','enzyme','parallel'),('research','enzyme','serial')]:
        execute(ROOT/f'tools/bench/protocols/posteriordb-native-{name}-v3.json',prepared,output/name,backend,execution)
    write_new(output/'complete.json',dict(schema_version=1,purpose='descriptive',
        runs={name:file_hash(output/name/'run/index.json') for name in ('pilot','broad','shared-stan','parallel','research')}))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='alea-bench native')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('build'); p.add_argument('output', type=Path); p.add_argument('--models', type=Path, required=True)
    p = commands.add_parser('conform'); p.add_argument('prepared', type=Path); p.add_argument('output', type=Path)
    p = commands.add_parser('campaign'); p.add_argument('output', type=Path); p.add_argument('--models', type=Path, required=True)
    p = commands.add_parser('run'); p.add_argument('protocol', type=Path); p.add_argument('prepared', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--backend', choices=('enzyme', 'bridgestan'), default='enzyme'); p.add_argument('--execution', choices=('serial', 'parallel'), default='serial')
    args = parser.parse_args(argv)
    if args.command == 'build': build(args.output, args.models)
    elif args.command == 'conform': conform(args.prepared, args.output)
    elif args.command == 'campaign': campaign(args.output,args.models)
    else: execute(args.protocol, args.prepared, args.output, args.backend, args.execution)
