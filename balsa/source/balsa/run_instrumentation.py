"""Append-only phase telemetry; optional, without changing training parameters."""
import contextlib
import functools
import json
import os
import time


def event(stage, kind, **extra):
    path = os.environ.get('BALSA_PHASE_LOG')
    if path:
        record = dict(stage=stage, event=kind, monotonic=time.monotonic(),
                      timestamp=time.time(), pid=os.getpid(), **extra)
        with open(path, 'a') as handle:
            handle.write(json.dumps(record) + '\n')


@contextlib.contextmanager
def phase(stage, **extra):
    started = time.monotonic()
    event(stage, 'START', **extra)
    success = False
    try:
        yield
        success = True
    finally:
        event(stage, 'END', seconds=time.monotonic() - started,
              completed=success, **extra)


def measure_phase(stage):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            iteration = getattr(args[0], 'curr_value_iter', None) if args else None
            with phase(stage, value_iter=iteration):
                return fn(*args, **kwargs)
        return wrapped
    return decorate
