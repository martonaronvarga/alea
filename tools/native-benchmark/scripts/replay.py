#!/usr/bin/env python3
"""Rebuild a summary to a NEW file without altering historical journals."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

import bench
import compare
import kernel_compare


def journal(path):
    """Only a torn final record is recoverable; malformed complete rows fail."""
    if not path.exists():
        return []
    # Only LF terminates a journal record. Never skip malformed interior lines.
    lines = path.read_bytes().split(b'\n')
    return [bench.read_metrics(line) for line in lines[:-1]]


def rebuild(directory, kind):
    if kind == 'conformance':
        return bench.conformance_summary(bench.read_metrics((directory/'manifest.json').read_text()),
                                         bench.read_metrics((directory/'report.json').read_text()))
    plan = json.loads((directory/('plan.json' if kind == 'paired' else 'manifest.json')).read_text())
    rows = journal(directory/'records.jsonl')
    def unique(records, keys):
        identities = [tuple(r[k] for k in keys) for r in records]
        if len(identities) != len(set(identities)):
            raise ValueError('duplicate journal identity')
    keys = ['case','engine','rep'] + (['mode'] if kind == 'paired' else ['size'] if kind == 'kernel' else [])
    unique(rows, keys)
    if kind == 'stan':
        return bench.summarize(rows,plan)
    decisions = journal(directory/'validation.jsonl')
    unique(decisions, ['case','rep', 'mode' if kind == 'paired' else 'size'])
    if kind == 'paired':
        return compare.summarize(rows,plan,decisions)
    return kernel_compare.summarize(rows,plan['cases'],decisions,plan['reps'])


# Generalized from the existing PosteriorDB segmented packer. A capsule seals
# bytes, not scientific truth; the source-bound readers still own admission.
class Segments(io.RawIOBase):
    def __init__(self, directory, limit):
        self.directory, self.limit = directory, limit
        self.stream = None
        self.size = 0
        self.paths = []

    def writable(self):
        return True

    def write(self, data):
        total = len(data)
        view = memoryview(data)
        while view:
            if self.stream is None or self.size == self.limit:
                if self.stream: self.stream.close()
                path = self.directory/f'evidence.tar.gz.part{len(self.paths)+1:03d}'
                self.paths.append(path)
                self.stream, self.size = path.open('xb'), 0
            n = min(len(view), self.limit-self.size)
            self.stream.write(view[:n])
            self.size += n
            view = view[n:]
        return total

    def finish(self):
        if self.stream: self.stream.close()


def relative(name):
    if not isinstance(name, str):
        raise ValueError('unsafe capsule path')
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts
            or not path.parts or path.as_posix() != name or '\\' in name):
        raise ValueError('unsafe capsule path')
    return path


def pack(directory, output, limit=32*1024*1024):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if not directory.is_dir() or output.is_relative_to(directory) or not 1 <= limit <= 64*1024*1024:
        raise ValueError('invalid capsule source/destination/segment size')
    paths = sorted(directory.rglob('*'))
    if any(p.is_symlink() or not (p.is_dir() or p.is_file()) for p in paths):
        raise ValueError('capsule source contains symlink or special file')
    files = {p.relative_to(directory).as_posix(): dict(bytes=p.stat().st_size, sha256=bench.sha(p))
             for p in paths if p.is_file()}
    if not files: raise ValueError('empty evidence capsule')
    if any(entry['bytes'] > 1024*1024*1024 for entry in files.values()):
        raise ValueError('capsule member exceeds 1 GiB limit')
    output.mkdir(parents=True, exist_ok=False)
    segments = Segments(output, limit)
    try:
        with gzip.GzipFile(fileobj=segments, mode='wb', filename='', mtime=0, compresslevel=1) as zipped, tarfile.open(fileobj=zipped, mode='w|') as archive:
            for name, entry in files.items():
                relative(name)
                path = directory/name
                item = tarfile.TarInfo('evidence/'+name)
                item.size, item.mode, item.mtime = entry['bytes'], 0o644, 0
                with path.open('rb') as stream: archive.addfile(item, stream)
                if path.stat().st_size != entry['bytes'] or bench.sha(path) != entry['sha256']:
                    raise ValueError('evidence source changed during packing')
    finally:
        segments.finish()
    index = dict(schema_version=1, format='alea-evidence-capsule', files=files,
                 parts=[dict(name=p.name, bytes=p.stat().st_size, sha256=bench.sha(p)) for p in segments.paths])
    if len(index['parts']) > 999:
        raise ValueError('capsule exceeds 999 segment limit')
    with (output/'capsule.json').open('x') as stream:
        json.dump(index, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return index


def capsule_index(directory):
    path = directory/'capsule.json'
    if path.is_symlink():
        raise ValueError('capsule index must not be a symlink')
    index = bench.read_metrics(path.read_text())
    if (set(index) != {'schema_version','format','files','parts'} or type(index['schema_version']) is not int
            or index['schema_version'] != 1 or index['format'] != 'alea-evidence-capsule'
            or not isinstance(index['files'], dict) or not index['files']
            or not isinstance(index['parts'], list) or not 1 <= len(index['parts']) <= 999):
        raise ValueError('invalid capsule index')
    def entry(value, maximum):
        if (not isinstance(value, dict) or set(value) != {'bytes','sha256'} or type(value['bytes']) is not int
                or not 0 <= value['bytes'] <= maximum or not isinstance(value['sha256'], str)
                or not re.fullmatch('[0-9a-f]{64}', value['sha256'])):
            raise ValueError('invalid capsule entry')
    for name, value in index['files'].items():
        relative(name); entry(value, 1024*1024*1024)
    for i, part in enumerate(index['parts']):
        if not isinstance(part, dict) or part.get('name') != f'evidence.tar.gz.part{i+1:03d}':
            raise ValueError('invalid capsule segment')
        entry({k:v for k,v in part.items() if k != 'name'}, 64*1024*1024)
        if part['bytes'] == 0: raise ValueError('empty capsule segment')
    return index


def unpack(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    index = capsule_index(directory)
    for part in index['parts']:
        path = directory/part['name']
        if path.is_symlink() or path.stat().st_size != part['bytes'] or bench.sha(path) != part['sha256']:
            raise ValueError('capsule segment checksum mismatch')
    output.mkdir(parents=True, exist_ok=False)
    combined = output/'archive.tar.gz'
    with combined.open('xb') as stream:
        for part in index['parts']:
            with (directory/part['name']).open('rb') as source: shutil.copyfileobj(source, stream)
    seen = set()
    with tarfile.open(combined, 'r:gz') as archive:
        for item in archive:
            relative(item.name)
            if not item.isfile() or not item.name.startswith('evidence/'):
                raise ValueError('invalid capsule archive member')
            name = item.name[len('evidence/'):]
            if name not in index['files'] or name in seen or item.size != index['files'][name]['bytes']:
                raise ValueError('unsealed/duplicate capsule archive member')
            target = output/'evidence'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(item) as source, target.open('xb') as stream:
                shutil.copyfileobj(source, stream)
            if bench.sha(target) != index['files'][name]['sha256']:
                raise ValueError('capsule member checksum mismatch')
            seen.add(name)
    if seen != set(index['files']): raise ValueError('incomplete capsule archive')
    combined.unlink() # Only our verified temporary combined archive, not evidence.
    return output/'evidence'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['stan','paired','kernel','conformance','pack','unpack'])
    parser.add_argument('directory',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    if args.kind == 'pack':
        pack(args.directory, args.output)
        return
    if args.kind == 'unpack':
        unpack(args.directory, args.output)
        return
    result = rebuild(args.directory,args.kind)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
