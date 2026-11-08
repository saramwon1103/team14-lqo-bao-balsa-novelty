#!/usr/bin/env python3
"""Independent systemd deadline backstop for the entire pilot slice, including eval."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    state = json.loads((args.run_dir / 'resource_summary.json').read_text())
    if state['status'] != 'RUNNING':
        raise RuntimeError('No active pilot to protect')
    slice_name = Path(state['cgroup']).name
    pid = int(slice_name[:-6].rsplit('-', 1)[1])
    # /proc starttime is earlier than the pipeline clock: conservative bound.
    fields = Path('/proc/{}/stat'.format(pid)).read_text().rsplit(') ', 1)[1].split()
    process_started = int(fields[19]) / os.sysconf('SC_CLK_TCK')
    remaining = max(1, int(7140 - (time.monotonic() - process_started)))
    guard = slice_name[:-6] + '-aggregate-deadline'
    command = ['systemd-run', '--user', '--unit=' + guard,
               '--on-active={}s'.format(remaining), '--timer-property=AccuracySec=1s',
               '/usr/bin/systemctl', '--user', 'stop', slice_name]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, check=True)
    (args.run_dir / 'aggregate_deadline_guard.json').write_text(json.dumps(dict(
        timer=guard + '.timer', slice=slice_name, remaining_seconds_when_installed=remaining,
        stop_target_seconds_after_supervisor_process_start=7140,
        service_stop_grace_seconds=25, purpose='aggregate wall-clock guard across train and fresh eval',
        systemd_output=result.stdout), indent=2))
    print(result.stdout)


if __name__ == '__main__':
    main()
