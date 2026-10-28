#!/usr/bin/env python3
"""Kiểm tra môi trường trước khi chạy Balsa chính thức.

Không tạo dữ liệu và không sửa PostgreSQL.  Mật khẩu, nếu cần, được lấy từ
cơ chế chuẩn của psql như PGPASSFILE hoặc lời nhắc tương tác.
"""

from __future__ import print_function

import argparse
import os
import shutil
import subprocess
import sys

try:
    import pkg_resources
except ImportError:
    pkg_resources = None

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    try:
        import importlib_metadata
    except ImportError:
        importlib_metadata = None


PINNED_PACKAGES = [
    ('ray', '1.9.2'),
    ('pytorch-lightning', '0.9.0'),
    ('torch', '1.4.0'),
    ('psycopg2-binary', '2.8.5'),
    ('absl-py', '0.9.0'),
    ('numpy', '1.18.1'),
    ('aiohttp', '3.7.4'),
    ('aioredis', '1.3.1'),
    ('grpcio', '1.43.0'),
]


class Checks(object):

    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def ok(self, message):
        print('[OK]   ' + message)

    def fail(self, message):
        self.failures += 1
        print('[FAIL] ' + message)

    def warn(self, message):
        self.warnings += 1
        print('[WARN] ' + message)


def command_output(command):
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return process.returncode, process.stdout.strip(), process.stderr.strip()


def psql_command(args):
    command = ['psql', '-X', '-v', 'ON_ERROR_STOP=1', '-At', '-d', args.db]
    if args.host:
        command.extend(['-h', args.host])
    if args.port:
        command.extend(['-p', str(args.port)])
    if args.user:
        command.extend(['-U', args.user])
    return command


def check_python(checks):
    version = '{}.{}.{}'.format(*sys.version_info[:3])
    if sys.version_info[:2] == (3, 7):
        checks.ok('Python {} đúng với môi trường tái lập của dự án'.format(version))
    else:
        checks.fail(
            'Python {} không phải Python 3.7 mà Balsa gốc yêu cầu'.format(version))

    for package, expected in PINNED_PACKAGES:
        try:
            if pkg_resources is not None:
                installed = pkg_resources.get_distribution(package).version
            elif importlib_metadata is not None:
                installed = importlib_metadata.version(package)
            else:
                raise LookupError
        except Exception:
            checks.fail('Thiếu gói {}=={}'.format(package, expected))
            continue
        if installed == expected:
            checks.ok('{}=={}'.format(package, installed))
        else:
            checks.fail('{}=={}, cần {}'.format(package, installed, expected))


def check_repository(checks, repo_root):
    required = [
        'run.py',
        'experiments.py',
        'requirements.txt',
        os.path.join('balsa', 'structured_logging.py'),
    ]
    for relative in required:
        path = os.path.join(repo_root, relative)
        if os.path.isfile(path):
            checks.ok('Có {}'.format(relative))
        else:
            checks.fail('Thiếu {}'.format(relative))

    query_dir = os.path.join(repo_root, 'queries', 'join-order-benchmark')
    queries = []
    if os.path.isdir(query_dir):
        queries = [name for name in os.listdir(query_dir) if name.endswith('.sql')]
    if len(queries) == 113:
        checks.ok('Có đủ 113 truy vấn JOB')
    else:
        checks.fail('Số truy vấn JOB là {}, cần 113'.format(len(queries)))


def check_postgres(checks, args):
    if shutil.which('psql') is None:
        checks.fail('Không tìm thấy lệnh psql')
        return
    checks.ok('Đã tìm thấy psql')

    command = psql_command(args)
    code, output, error = command_output(
        command + ['-c', 'SHOW server_version;'])
    if code != 0:
        checks.fail('Không kết nối được cơ sở dữ liệu {}: {}'.format(
            args.db, error.splitlines()[-1] if error else 'không rõ lỗi'))
        return
    version = output.splitlines()[-1] if output else ''
    if version.startswith('12.5'):
        checks.ok('PostgreSQL {}'.format(version))
    else:
        checks.fail('PostgreSQL {}, bản tái lập chính thức dùng 12.5'.format(version))

    sql_checks = [
        ('21 bảng JOB',
         "SELECT count(*) FROM information_schema.tables "
         "WHERE table_schema='public' AND table_type='BASE TABLE';",
         '21'),
        ('2.528.312 bản ghi trong title',
         'SELECT count(*) FROM title;',
         '2528312'),
    ]
    for label, sql, expected in sql_checks:
        code, output, error = command_output(command + ['-c', sql])
        if code == 0 and output.splitlines()[-1:] == [expected]:
            checks.ok(label)
        elif code == 0:
            checks.fail('{}: nhận {}, cần {}'.format(label, output, expected))
        else:
            checks.fail('{}: {}'.format(
                label, error.splitlines()[-1] if error else 'truy vấn lỗi'))

    code, _, error = command_output(
        command + ['-c', "LOAD 'pg_hint_plan'; SELECT 1;"])
    if code == 0:
        checks.ok('Nạp được pg_hint_plan')
    else:
        checks.fail('Không nạp được pg_hint_plan: {}'.format(
            error.splitlines()[-1] if error else 'không rõ lỗi'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.environ.get('BALSA_DB', 'imdbload'))
    parser.add_argument('--host', default=os.environ.get('PGHOST'))
    parser.add_argument('--port', default=os.environ.get('PGPORT'))
    parser.add_argument('--user', default=os.environ.get('PGUSER'))
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checks = Checks()
    print('=== KIỂM TRA MÔI TRƯỜNG BALSA ===')
    check_repository(checks, repo_root)
    check_python(checks)
    check_postgres(checks, args)
    print('=== KẾT QUẢ: {} lỗi, {} cảnh báo ==='.format(
        checks.failures, checks.warnings))
    return 1 if checks.failures else 0


if __name__ == '__main__':
    sys.exit(main())
