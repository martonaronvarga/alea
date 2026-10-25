"""One pinned worker process per chain; pooled diagnostics run after merging."""
import concurrent.futures
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from .manifest import ROOT, write_new
from .runner import environment


def run(job):
    directory = Path(job['directory']); start = time.perf_counter_ns()

    def chain(i):
        folder = directory/f'chain-{i}'; folder.mkdir()
        task = dict(job, directory=str(folder), seeds=[job['seeds'][i]], initials=[job['initials'][i]])
        write_new(folder/'job.json', task)
        command = [shutil.which('time'), '-f', '%M', '-o', str(folder/'rss.txt'),
                   shutil.which('taskset'), '-c', str(job['parallel_cpus'][i]),
                   *job['parallel_command'], str(folder/'job.json')]
        with (folder/'process.log').open('x') as log:
            began=time.perf_counter_ns()
            # Inherit the enclosing measured job's process group. Its pidfd
            # deadline kills the entire cohort, including time/taskset children.
            # A new child session here would escape that cancellation boundary.
            child=subprocess.Popen(command,cwd=ROOT,env=environment(ROOT),stdout=log,stderr=subprocess.STDOUT)
            code=child.wait();elapsed=time.perf_counter_ns()-began
            process=dict(returncode=code,wall_nanoseconds=elapsed,wall_seconds=elapsed/1e9,
                         timed_out=False,clock='perf_counter_ns + blocking child wait; deadline owned by enclosing job')
        write_new(folder/'process.json', process)
        return folder, process

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(job['seeds'])) as pool:
        results = list(pool.map(chain, range(len(job['seeds']))))
    phases = []; counts = {}; backends = set(); peaks = []; failures = []
    for i, (folder, process) in enumerate(results):
        worker_path = folder/'worker.json'
        if not worker_path.exists(): worker_path = folder/'progress-worker.json'
        worker = json.loads(worker_path.read_text()) if worker_path.exists() else {}
        for phase in worker.get('phases', []): phases.append(dict(phase, chain=i))
        backends.add(worker.get('backend', 'unavailable'))
        for key, value in worker.get('work', {}).items():
            if type(value) is int: counts[key] = counts.get(key, 0)+value
        rss = (folder/'rss.txt').read_text().splitlines()
        if rss and rss[-1].isdigit(): peaks.append(int(rss[-1]))
        if process['returncode'] or process['timed_out']: failures.append(dict(chain=i, **process))
    for name in ('draws.csv', 'constrained.csv'):
        with (directory/name).open('x', newline='') as out:
            writer = csv.writer(out); header = None
            for i, (folder, _) in enumerate(results):
                if not (folder/name).exists(): continue
                with (folder/name).open(newline='') as source:
                    reader = csv.reader(source); actual = next(reader)
                    if header is None: header = actual; writer.writerow(header)
                    if actual != header: raise ValueError('parallel coordinate header mismatch')
                    for row in reader: row[0] = i; writer.writerow(row)
    with tarfile.open(directory/'chain-artifacts.tar.gz', 'x:gz') as archive:
        for folder, _ in results: archive.add(folder, arcname=folder.name)
    record = dict(schema_version=2, status='failed' if failures else 'completed',
                  failure=failures or None, phases=phases, work=counts,
                  backend='; '.join(sorted(backends)), execution='parallel process per chain',
                  jit_seconds=sum(json.loads((f/'worker.json').read_text()).get('jit_seconds', 0.) for f, _ in results if (f/'worker.json').exists()),
                  worker_seconds=(time.perf_counter_ns()-start)/1e9,
                  chain_peak_rss_sum_upper_bound_kib=sum(peaks) if len(peaks)==len(results) else None,
                  memory_scope='sum of individual chain process peaks is an upper bound, not simultaneous aggregate RSS')
    write_new(directory/'worker.json', record)
    if failures: raise RuntimeError('one or more parallel chains failed; partial artifacts retained')


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))
