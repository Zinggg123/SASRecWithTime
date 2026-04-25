import argparse
import csv
import hashlib
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def sanitize_name(raw):
    safe = []
    for ch in raw:
        if ch.isalnum() or ch in ('-', '_'):
            safe.append(ch)
        else:
            safe.append('_')
    return ''.join(safe)


def split_common_args(common_args):
    tokens = []
    for fragment in common_args:
        tokens.extend(shlex.split(fragment))
    return tokens


def parse_config_line(raw_line):
    stripped = raw_line.strip()
    if not stripped or stripped.startswith('#'):
        return None

    normalized = stripped.replace(',', ' ')
    tokens = shlex.split(normalized)
    config_args = []
    config_values = {}

    for token in tokens:
        if token.startswith('--'):
            config_args.append(token)
            key_value = token[2:]
        elif '=' in token:
            config_args.append(f'--{token}')
            key_value = token
        else:
            config_args.append(token)
            key_value = None

        if key_value and '=' in key_value:
            key, value = key_value.split('=', 1)
            config_values[key.strip().lower()] = value.strip()

    if not config_args:
        return None

    label_keys = {'name', 'config_name', 'cfg_name'}
    filtered_args = []
    for token in config_args:
        key = None
        if token.startswith('--') and '=' in token:
            key = token[2:].split('=', 1)[0].strip().lower()
        elif '=' in token:
            key = token.split('=', 1)[0].strip().lower()
        if key in label_keys:
            continue
        filtered_args.append(token)

    config_name = (
        config_values.get('name')
        or config_values.get('config_name')
        or config_values.get('cfg_name')
    )
    if config_name is None:
        digest = hashlib.md5(stripped.encode('utf-8')).hexdigest()[:8]
        config_name = f'line_{digest}'

    return {
        'raw_line': stripped,
        'config_args': filtered_args,
        'config_name': config_name,
        'config_hash': hashlib.md5(stripped.encode('utf-8')).hexdigest()[:12],
    }


def read_config_doc(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    tasks = []
    for line_no, raw_line in enumerate(lines, start=1):
        parsed = parse_config_line(raw_line)
        if parsed is None:
            continue
        parsed['line_no'] = line_no
        tasks.append(parsed)
    return tasks


def build_command(python_dir, dataset, train_dir, common_args, config_args):
    cmd = [sys.executable, '-u', 'main.py', f'--dataset={dataset}', f'--train_dir={train_dir}']
    cmd.extend(common_args)
    cmd.extend(config_args)
    return cmd


def run_one_task(python_dir, output_root, dataset, train_dir, common_args, task):
    run_label = f"line{task['line_no']:03d}_{sanitize_name(task['config_name'])}_{task['config_hash']}"
    run_dir = os.path.join(output_root, run_label)
    os.makedirs(run_dir, exist_ok=True)

    stdout_path = os.path.join(run_dir, 'stdout.log')
    stderr_path = os.path.join(run_dir, 'stderr.log')
    command_path = os.path.join(run_dir, 'command.txt')

    cmd = build_command(python_dir, dataset, train_dir, common_args, task['config_args'])
    with open(command_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(shlex.quote(part) for part in cmd))
        f.write('\n')

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    start_time = time.time()
    try:
        with open(stdout_path, 'w', encoding='utf-8') as out, open(stderr_path, 'w', encoding='utf-8') as err:
            proc = subprocess.run(cmd, cwd=python_dir, stdout=out, stderr=err, env=env)
        returncode = proc.returncode
        status = 'ok' if returncode == 0 else f'failed_rc_{returncode}'
        error_message = ''
    except Exception as exc:
        returncode = None
        status = 'failed_exception'
        error_message = f'{type(exc).__name__}: {exc}'
        with open(stderr_path, 'a', encoding='utf-8') as err:
            err.write(error_message + '\n')

    duration_sec = time.time() - start_time
    return {
        'line_no': task['line_no'],
        'config_name': task['config_name'],
        'config_hash': task['config_hash'],
        'raw_line': task['raw_line'],
        'run_dir': run_dir,
        'stdout_log': stdout_path,
        'stderr_log': stderr_path,
        'command': ' '.join(shlex.quote(part) for part in cmd),
        'returncode': returncode,
        'status': status,
        'error_message': error_message,
        'duration_sec': round(duration_sec, 3),
    }


def write_summary_csv(path, rows):
    fieldnames = [
        'line_no',
        'config_name',
        'config_hash',
        'status',
        'returncode',
        'duration_sec',
        'run_dir',
        'stdout_log',
        'stderr_log',
        'command',
        'raw_line',
        'error_message',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Read one config per line from a document and run SASRec with limited concurrency.')
    parser.add_argument('--config_doc', required=True, help='Path to the config document, one config per line.')
    parser.add_argument('--dataset', default='ml-1m', help='Base dataset passed to main.py.')
    parser.add_argument('--train_dir', default='doc_run', help='Base train_dir passed to main.py.')
    parser.add_argument('--output_root', default='doc_run_outputs', help='Directory used to store run outputs and summary.')
    parser.add_argument('--max_workers', type=int, default=2, help='Maximum number of configs to run in parallel.')
    parser.add_argument('--common_arg', action='append', default=[], help='Shared argument fragment appended before each line config, e.g. "--device=cuda --maxlen=200".')

    args = parser.parse_args()
    if args.max_workers < 1:
        raise ValueError('--max_workers must be >= 1')

    python_dir = os.path.dirname(os.path.abspath(__file__))
    config_doc_path = os.path.abspath(args.config_doc)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    tasks = read_config_doc(config_doc_path)
    if not tasks:
        print(f'No runnable config lines found in {config_doc_path}')
        return

    common_args = split_common_args(args.common_arg)
    total_tasks = len(tasks)
    max_workers = min(args.max_workers, total_tasks)
    print(f'Read {total_tasks} configs from {config_doc_path}')
    print(f'Running with max_workers={max_workers}')

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_one_task, python_dir, output_root, args.dataset, args.train_dir, common_args, task): task
            for task in tasks
        }

        completed = 0
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    'line_no': task['line_no'],
                    'config_name': task['config_name'],
                    'config_hash': task['config_hash'],
                    'raw_line': task['raw_line'],
                    'run_dir': '',
                    'stdout_log': '',
                    'stderr_log': '',
                    'command': '',
                    'returncode': None,
                    'status': 'failed_future_exception',
                    'error_message': f'{type(exc).__name__}: {exc}',
                    'duration_sec': None,
                }

            completed += 1
            results.append(result)
            print(
                f"[{completed}/{total_tasks}] line {result['line_no']} {result['config_name']} -> {result['status']} "
                f"(rc={result['returncode']}, {result['duration_sec']}s)",
                flush=True,
            )

    summary_path = os.path.join(output_root, 'summary.csv')
    results.sort(key=lambda row: row['line_no'])
    write_summary_csv(summary_path, results)
    print(f'Summary written to {summary_path}')


if __name__ == '__main__':
    main()