# Copyright 2022 The Balsa Authors.
# Local extension Copyright 2026 The Balsa Logging Extension Authors.
#
# Licensed under the Apache License, Version 2.0.
"""Structured, human-readable experiment logs for official Balsa runs.

This module deliberately uses only the Python standard library so its output
can be tested without importing the legacy Balsa ML dependency stack.
"""

from __future__ import print_function

import csv
import json
import os
import shutil
if __package__:
    from .explain_parser import normalize_explain
else:
    from explain_parser import normalize_explain
from datetime import datetime
from datetime import timezone


RESULT_FIELDS = [
    'value_iter',
    'split',
    'model',
    'query',
    'balsa_planning_ms',
    'balsa_execution_ms',
    'balsa_end_to_end_ms',
    'postgres_planning_ms',
    'postgres_execution_ms',
    'postgres_end_to_end_ms',
    'speedup_vs_postgres',
    'status',
]
CONTRACT_FIELDS = '''run_id protocol split_id fold_id seed query_id repeat_id checkpoint_sha256 split_sha256 postgres_hinted_planning_ms execution_ms end_to_end_ms wall_clock_ms postgres_baseline_planning_ms postgres_baseline_execution_ms postgres_baseline_end_to_end_ms speedup_execution speedup_end_to_end timeout_ms result_source hint_respected hinted_sql_path explain_json_path error_message'''.split()
RESULT_FIELDS += [f for f in CONTRACT_FIELDS if f not in RESULT_FIELDS]


def _query_label(query_name):
    value = str(query_name)
    if value.endswith('.sql'):
        value = value[:-4]
    return value if value.startswith('q') else 'q' + value


def _number(value, digits=3):
    if value is None or value == '':
        return 'N/A'
    try:
        return ('{0:.' + str(digits) + 'f}').format(float(value))
    except (TypeError, ValueError):
        return str(value)


def extract_postgres_metrics(result):
    """Returns total cost, plan rows, planning ms and execution ms."""
    if hasattr(result, 'result'):
        result = result.result
    if not result:
        return {
            'cost': None,
            'cardinality': None,
            'planning_ms': None,
            'execution_ms': None,
        }
    try:
        payload = normalize_explain(result)
        plan = payload['Plan']
        return {
            'cost': plan.get('Total Cost'),
            'cardinality': plan.get('Plan Rows'),
            'planning_ms': payload.get('Planning Time'),
            'execution_ms': payload.get('Execution Time'),
        }
    except (IndexError, KeyError, TypeError, ValueError):
        return {
            'cost': None,
            'cardinality': None,
            'planning_ms': None,
            'execution_ms': None,
        }


class StructuredExperimentLogger(object):
    """Collects one Balsa value iteration and writes the requested format."""

    def __init__(self, output_dir, split_name, model_name='Balsa'):
        self.output_dir = os.path.abspath(output_dir)
        self.split_name = split_name
        self.model_name = model_name
        self.iterations_dir = os.path.join(self.output_dir, 'iterations')
        self.checkpoints_dir = os.path.join(self.output_dir, 'checkpoints')
        self.master_log = os.path.join(self.output_dir, 'experiment.log')
        self.results_csv = os.path.join(self.output_dir, 'balsa_results.csv')
        self.baseline_csv = os.path.join(self.output_dir, 'postgres_baseline.csv')
        if os.path.exists(self.master_log) or os.path.exists(self.results_csv):
            raise FileExistsError('Refusing to overwrite an existing structured run')
        self.current = None
        self.metadata = {}
        os.makedirs(self.iterations_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def start_iteration(self, value_iter):
        self.current = {
            'value_iter': int(value_iter),
            'training_rows': [],
            'encoding_rows': [],
            'training': {
                'status': 'NOT_STARTED',
                'time_ms': None,
                'model_state': 'N/A',
            },
            'evaluation_rows': [],
        }

    def _require_current(self):
        if self.current is None:
            raise RuntimeError('start_iteration() must be called first')

    def record_encoding(self, query_name, planning_ms=None):
        self._require_current()
        label = _query_label(query_name)
        if any(row['query'] == label for row in self.current['encoding_rows']):
            return
        self.current['encoding_rows'].append({
            'query': label,
            'planning_ms': planning_ms,
        })

    def record_training_plan(self,
                             query_name,
                             result,
                             execution_ms,
                             status='COMPLETED', result_source=None):
        self._require_current()
        metrics = extract_postgres_metrics(result)
        self.current['training_rows'].append({
            'query': _query_label(query_name),
            'plan_collected': 'YES' if status == 'COMPLETED' else 'NO',
            'cost': metrics['cost'],
            'cardinality': metrics['cardinality'],
            'execution_ms': execution_ms,
            'status': status,
            'result_source': result_source,
        })

    def record_training(self, time_ms, model_state, status='COMPLETED'):
        self._require_current()
        self.current['training'] = {
            'status': status,
            'time_ms': time_ms,
            'model_state': model_state,
        }

    def record_evaluation(self,
                          query_name,
                          planning_ms,
                          execution_ms,
                          postgres_planning_ms,
                          postgres_execution_ms,
                          status='COMPLETED', postgres_hinted_planning_ms=None,
                          result_source=None, hinted_sql_path=None,
                          explain_json_path=None, timeout_ms=None):
        self._require_current()
        balsa_end_to_end = None
        postgres_end_to_end = None
        speedup = None
        if (status == 'COMPLETED' and planning_ms is not None and execution_ms is not None
                and execution_ms >= 0 and postgres_hinted_planning_ms is not None):
            balsa_end_to_end = float(planning_ms) + float(execution_ms) + float(postgres_hinted_planning_ms)
        if postgres_execution_ms is not None:
            postgres_end_to_end = float(postgres_execution_ms)
            if postgres_planning_ms is not None:
                postgres_end_to_end += float(postgres_planning_ms)
        if balsa_end_to_end and postgres_end_to_end is not None:
            speedup = postgres_end_to_end / balsa_end_to_end
        self.current['evaluation_rows'].append({
            'query': _query_label(query_name),
            'balsa_planning_ms': planning_ms,
            'balsa_execution_ms': execution_ms,
            'balsa_end_to_end_ms': balsa_end_to_end,
            'postgres_planning_ms': postgres_planning_ms,
            'postgres_execution_ms': postgres_execution_ms,
            'postgres_end_to_end_ms': postgres_end_to_end,
            'speedup_vs_postgres': speedup,
            'status': status,
            'postgres_hinted_planning_ms': postgres_hinted_planning_ms,
            'execution_ms': execution_ms,
            'end_to_end_ms': balsa_end_to_end,
            'postgres_baseline_planning_ms': postgres_planning_ms,
            'postgres_baseline_execution_ms': postgres_execution_ms,
            'postgres_baseline_end_to_end_ms': postgres_end_to_end,
            'speedup_execution': (float(postgres_execution_ms) / execution_ms
                if status == 'COMPLETED' and execution_ms and postgres_execution_ms is not None else None),
            'speedup_end_to_end': speedup,
            'result_source': result_source,
            'hinted_sql_path': hinted_sql_path,
            'explain_json_path': explain_json_path,
            'timeout_ms': timeout_ms,
        })

    def record_baseline(self, query_name, result, status='COMPLETED'):
        metrics = extract_postgres_metrics(result)
        row = {
            'query': _query_label(query_name),
            'cost': metrics['cost'],
            'cardinality': metrics['cardinality'],
            'planning_ms': metrics['planning_ms'],
            'execution_ms': metrics['execution_ms'],
            'end_to_end_ms': (
                float(metrics['planning_ms']) + float(metrics['execution_ms'])
                if metrics['planning_ms'] is not None and metrics['execution_ms'] is not None
                else None
            ),
            'status': status,
        }
        exists = os.path.exists(self.baseline_csv)
        with open(self.baseline_csv, 'a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def mirror_checkpoint(self, source_path, value_iter):
        """Copies the latest official checkpoint into the log directory."""
        self._require_current()
        destination = os.path.join(self.checkpoints_dir, 'latest.pt')
        shutil.copy2(source_path, destination + '.tmp')
        os.replace(destination + '.tmp', destination)
        metadata_path = os.path.join(self.checkpoints_dir, 'latest.json')
        with open(metadata_path, 'w') as handle:
            json.dump({
                'value_iter': int(value_iter),
                'split': self.split_name,
                'model': self.model_name,
                'source_path': os.path.abspath(source_path),
            }, handle, indent=2, sort_keys=True)
        return destination

    def _render(self):
        self._require_current()
        current = self.current
        lines = [
            '===== EXPERIMENT START =====',
            '',
            'Split: {}'.format(self.split_name),
            'Model: {}'.format(self.model_name),
            'Fold: {}'.format(self.metadata.get('fold_id') or 'N/A'),
            'Seed: {}'.format(self.metadata.get('seed') if self.metadata.get('seed') is not None else 'N/A'),
            'Value iteration: {}'.format(current['value_iter']),
            '',
            '[1] TRAINING DATA GENERATION',
        ]
        if current['training_rows']:
            for row in current['training_rows']:
                lines.extend([
                    'Train query: {}'.format(row['query']),
                    'Plan collected: {}'.format(row['plan_collected']),
                    'Cost: {}'.format(_number(row['cost'])),
                    'Cardinality: {}'.format(_number(row['cardinality'], 0)),
                    'Execution time: {} ms'.format(_number(row['execution_ms'])),
                    'Status: {}'.format(row['status']),
                    'Result source: {}'.format(row.get('result_source') or 'N/A'),
                    '',
                ])
        else:
            lines.extend(['Train query: N/A', 'Plan collected: NO', 'Status: NO_DATA', ''])

        lines.append('[2] QUERY & PLAN ENCODING')
        if current['encoding_rows']:
            for row in current['encoding_rows']:
                lines.extend([
                    'Query: {}'.format(row['query']),
                    'Query/Plan encoded: YES',
                    'Model input generated: YES',
                    '',
                ])
        else:
            lines.extend(['Query: N/A', 'Query/Plan encoded: NO', 'Model input generated: NO', ''])

        training = current['training']
        lines.extend([
            '[3] LQO TRAINING',
            'Model: {}'.format(self.model_name),
            'Split: {}'.format(self.split_name),
            'Training status: {}'.format(training['status']),
            'Training time: {} ms'.format(_number(training['time_ms'])),
            'Model/state: {}'.format(training['model_state']),
            '',
            '[4] LQO EVALUATION',
        ])
        if current['evaluation_rows']:
            for row in current['evaluation_rows']:
                lines.extend([
                    'Test query: {}'.format(row['query']),
                    '{} inference/planning: {} ms'.format(
                        self.model_name, _number(row['balsa_planning_ms'])),
                    'Execution time: {} ms'.format(_number(row['balsa_execution_ms'])),
                    'PostgreSQL hinted planning: {} ms'.format(_number(row.get('postgres_hinted_planning_ms'))),
                    'Plan: {}'.format(row.get('explain_json_path') or 'N/A'),
                    'End-to-End: {} ms'.format(_number(row['balsa_end_to_end_ms'])),
                    'Status: {}'.format(row['status']),
                    '',
                    '[POSTGRESQL]',
                    'Planning: {} ms'.format(_number(row['postgres_planning_ms'])),
                    'Execution: {} ms'.format(_number(row['postgres_execution_ms'])),
                    'End-to-End: {} ms'.format(_number(row['postgres_end_to_end_ms'])),
                    '',
                ])
        else:
            lines.extend([
                'Test query: N/A',
                '{} inference/planning: N/A ms'.format(self.model_name),
                'Execution time: N/A ms',
                'End-to-End: N/A ms',
                'Status: SKIPPED',
                '',
                '[POSTGRESQL]',
                'Planning: N/A ms',
                'Execution: N/A ms',
                'End-to-End: N/A ms',
                '',
            ])
        lines.extend(['===== EXPERIMENT END =====', ''])
        return '\n'.join(lines)

    def _append_results_csv(self):
        current = self.current
        exists = os.path.exists(self.results_csv)
        with open(self.results_csv, 'a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
            if not exists:
                writer.writeheader()
            for row in current['evaluation_rows']:
                payload = {
                    'value_iter': current['value_iter'],
                    'split': self.split_name,
                    'model': self.model_name,
                }
                payload.update(row)
                payload.update(run_id=os.path.basename(self.output_dir),
                               split_id=self.split_name, query_id=row['query'],
                               repeat_id=current['value_iter'])
                payload.update(self.metadata)
                writer.writerow({field: payload.get(field, '') for field in RESULT_FIELDS})

    def finish_iteration(self):
        self._require_current()
        text = self._render()
        value_iter = self.current['value_iter']
        iteration_path = os.path.join(
            self.iterations_dir,
            'value_iter_{:04d}.log'.format(value_iter),
        )
        if os.path.exists(iteration_path):
            raise FileExistsError(iteration_path)
        with open(iteration_path, 'w') as handle:
            handle.write(text)
        with open(self.master_log, 'a') as handle:
            handle.write(text)
        self._append_results_csv()

        metadata_path = os.path.join(
            self.iterations_dir,
            'value_iter_{:04d}.json'.format(value_iter),
        )
        payload = dict(self.current)
        payload['split'] = self.split_name
        payload['model'] = self.model_name
        payload['created_at'] = datetime.now(timezone.utc).isoformat().replace(
            '+00:00', 'Z')
        with open(metadata_path, 'w') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        self.current = None
        return iteration_path

    def archive_master_log(self):
        if not os.path.exists(self.master_log):
            return None
        archive_dir = os.path.join(self.output_dir, 'experiment_logs')
        os.makedirs(archive_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        destination = os.path.join(
            archive_dir,
            '{}_{}_{}.log'.format(timestamp, self.split_name, self.model_name),
        )
        shutil.copy2(self.master_log, destination)
        return destination
