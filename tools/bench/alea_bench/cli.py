"""Application entry points; the manifest and report library owns the contracts."""
import argparse
import json
from pathlib import Path
import sys
from . import schema
from .manifest import seal
from .prepare import prepare
from .runner import run
from .report import build_report, load_bundle, track, write_report


def main(argv=None):
    arguments=sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in ('pdb','native'):
        if arguments[0]=='native':
            from .native import main as descriptive
        else:
            from .pdb import main as descriptive
        try:descriptive(arguments[1:])
        except (schema.Invalid,OSError,RuntimeError,ValueError,KeyError) as error:
            print(f'alea-bench pdb: {error}',file=sys.stderr);return 1
        return 0
    parser=argparse.ArgumentParser(prog='alea-bench',description='Versioned, source-bound inference experiments. All output paths must be new.')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare',help='build pinned engines outside measured jobs');p.add_argument('protocol',type=Path);p.add_argument('directory',type=Path)
    p=commands.add_parser('reproduce',help='prepare all engines and regenerate a complete run from one sealed manifest');p.add_argument('manifest',type=Path);p.add_argument('directory',type=Path)
    p=commands.add_parser('seal',help='freeze source/data/policy/thresholds before running');p.add_argument('protocol',type=Path);p.add_argument('manifest',type=Path)
    p=commands.add_parser('run',help='fresh subprocesses, raw draws, failure records and report');p.add_argument('manifest',type=Path);p.add_argument('engines',type=Path);p.add_argument('directory',type=Path)
    p=commands.add_parser('report',help='regenerate a report from a complete immutable bundle');p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--require-quality',action='store_true')
    p=commands.add_parser('verify',help='check schema, digests and completeness including failed jobs');p.add_argument('directory',type=Path)
    p=commands.add_parser('track',help='host-separated uncertainty/regression observations');p.add_argument('directories',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':prepare(args.directory.resolve(),schema.protocol(schema.load(args.protocol))['engines'])
        elif args.command=='reproduce':
            from .manifest import verify_sources
            m=schema.manifest(schema.load(args.manifest));verify_sources(m)
            args.directory.mkdir(parents=True,exist_ok=False);prepare((args.directory/'preparation').resolve(),m['protocol']['engines']);run(args.manifest,(args.directory/'preparation/engines.json').resolve(),(args.directory/'run').resolve());write_report((args.directory/'run').resolve())
        elif args.command=='seal':seal(args.protocol,args.manifest)
        elif args.command=='run':run(args.manifest,args.engines,args.directory.resolve());r=write_report(args.directory.resolve());print(json.dumps(r['promotion'],indent=2))
        elif args.command=='verify':m,r=load_bundle(args.directory);print(f'{m["manifest_sha256"]}: {len(r)} planned jobs verified, failures retained')
        elif args.command=='report':
            report=build_report(args.directory)
            from .manifest import write_new
            write_new(args.output,report)
            if args.require_quality and not report['promotion']['correctness']:return 2
        elif args.command=='track':track(args.directories,args.output)
    except (schema.Invalid,OSError,RuntimeError,ValueError,KeyError) as error:
        print(f'alea-bench: {error}',file=sys.stderr);return 1
    return 0
