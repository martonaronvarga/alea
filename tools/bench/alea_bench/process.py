"""Kernel-notified child completion: no timeout-polling latency in measurements."""
import os
import select
import signal
import subprocess
import time


def isolated(command, *, cwd, env, log, timeout):
    """Measure fresh-process wall time with perf_counter_ns and Linux pidfd readiness.

    The descriptor wakes on exit. wait() only reaps an already exited child; no
    Popen.wait(timeout=...) exponential polling enters the measured interval.
    Deadline expiry kills the entire child session and reaps the direct child.
    """
    start=time.perf_counter_ns()
    process=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    descriptor=None;expired=False
    try:
        descriptor=os.pidfd_open(process.pid)
        ready,_,_=select.select([descriptor],[],[],timeout)
        expired=not ready
        if expired:os.killpg(process.pid,signal.SIGKILL)
        code=process.wait()
        elapsed=time.perf_counter_ns()-start
        return {'returncode':code,'wall_nanoseconds':elapsed,'wall_seconds':elapsed/1e9,'timed_out':expired,'clock':'perf_counter_ns + Linux pidfd exit readiness'}
    finally:
        if descriptor is not None:os.close(descriptor)
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
