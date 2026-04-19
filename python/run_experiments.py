import argparse
import csv
import hashlib
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from statistics import mean, pstdev


def bool_to_str(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'none'
    return str(value)


def sanitize_name(raw):
    safe = []
    for ch in raw:
        if ch.isalnum() or ch in ('-', '_'):
            safe.append(ch)
        else:
            safe.append('_')
    return ''.join(safe)


def non_empty_feature_subsets(prefix):
    # Three continuous-time features: N/C/R
    options = [
        ('N', {'normalized_gap': True, 'recent_compactness': False, 'recency_score': False}),
        ('C', {'normalized_gap': False, 'recent_compactness': True, 'recency_score': False}),
        ('R', {'normalized_gap': False, 'recent_compactness': False, 'recency_score': True}),
        ('NC', {'normalized_gap': True, 'recent_compactness': True, 'recency_score': False}),
        ('NR', {'normalized_gap': True, 'recent_compactness': False, 'recency_score': True}),
        ('CR', {'normalized_gap': False, 'recent_compactness': True, 'recency_score': True}),
        ('NCR', {'normalized_gap': True, 'recent_compactness': True, 'recency_score': True}),
    ]

    subsets = []
    for label, vals in options:
        subsets.append({
            f'{prefix}_use_normalized_gap': vals['normalized_gap'],
            f'{prefix}_use_recent_compactness': vals['recent_compactness'],
            f'{prefix}_use_recency_score': vals['recency_score'],
            f'{prefix}_subset_label': label,
        })
    return subsets


def build_command(dataset, train_dir, seed, config):
    cmd = [sys.executable, 'main.py', f'--dataset={dataset}', f'--train_dir={train_dir}', f'--seed={seed}']
    for key, value in config.items():
        if key.endswith('_subset_label'):
            continue
        cmd.append(f'--{key}={bool_to_str(value)}')
    return cmd


def parse_log(log_path):
    if not os.path.isfile(log_path):
        return None

    rows = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except OSError:
        return {
            'status': 'log_read_io_error',
            'mean_loss': None,
            'last_loss': None,
            'best_epoch': None,
            'best_val_ndcg': None,
            'best_val_hr': None,
            'test_ndcg_at_best_val': None,
            'test_hr_at_best_val': None,
        }

    numeric_rows = []
    for row in rows:
        epoch = row.get('epoch', '')
        if epoch.isdigit():
            numeric_rows.append(row)

    if not numeric_rows:
        return {
            'status': 'no_numeric_rows',
            'mean_loss': None,
            'last_loss': None,
            'best_epoch': None,
            'best_val_ndcg': None,
            'best_val_hr': None,
            'test_ndcg_at_best_val': None,
            'test_hr_at_best_val': None,
        }

    losses = []
    eval_rows = []
    for row in numeric_rows:
        loss_txt = row.get('loss', 'NA')
        if loss_txt != 'NA':
            try:
                losses.append(float(loss_txt))
            except ValueError:
                pass
        val_ndcg_txt = row.get('val_ndcg', 'NA')
        val_hr_txt = row.get('val_hr', 'NA')
        if val_ndcg_txt != 'NA' and val_hr_txt != 'NA':
            try:
                float(val_ndcg_txt)
                float(val_hr_txt)
                eval_rows.append(row)
            except ValueError:
                pass

    if eval_rows:
        # Keep consistent with main.py: update best when val_ndcg OR val_hr improves.
        best_val_ndcg_seen = float('-inf')
        best_val_hr_seen = float('-inf')
        best_row = None
        for row in eval_rows:
            cur_val_ndcg = float(row['val_ndcg'])
            cur_val_hr = float(row['val_hr'])
            if cur_val_ndcg > best_val_ndcg_seen or cur_val_hr > best_val_hr_seen:
                best_val_ndcg_seen = max(best_val_ndcg_seen, cur_val_ndcg)
                best_val_hr_seen = max(best_val_hr_seen, cur_val_hr)
                best_row = row

        best_epoch = int(best_row['epoch'])
        best_val_ndcg = float(best_row['val_ndcg'])
        best_val_hr = float(best_row['val_hr'])
        test_ndcg = float(best_row['test_ndcg'])
        test_hr = float(best_row['test_hr'])
    else:
        best_epoch = None
        best_val_ndcg = None
        best_val_hr = None
        test_ndcg = None
        test_hr = None

    return {
        'status': 'ok',
        'mean_loss': mean(losses) if losses else None,
        'last_loss': losses[-1] if losses else None,
        'best_epoch': best_epoch,
        'best_val_ndcg': best_val_ndcg,
        'best_val_hr': best_val_hr,
        'test_ndcg_at_best_val': test_ndcg,
        'test_hr_at_best_val': test_hr,
    }


def stable_config_id(stage, cfg_name, config):
    payload = stage + '|' + cfg_name + '|' + '|'.join(f'{k}={config[k]}' for k in sorted(config.keys()))
    digest = hashlib.md5(payload.encode('utf-8')).hexdigest()[:8]
    return f'{stage}_{sanitize_name(cfg_name)}_{digest}'


def make_failed_row(dataset, seed_index, seed, stage, cfg_name, config, config_id, train_dir, status, error_message):
    return {
        'row_type': 'run',
        'status': status,
        'log_parse_status': None,
        'dataset': dataset,
        'repeat': seed_index,
        'seed_index': seed_index,
        'stage': stage,
        'config_name': cfg_name,
        'config_id': config_id,
        'train_dir': train_dir,
        'seed': seed,
        'error_message': error_message,
        **flatten_config(config),
        'mean_loss': None,
        'last_loss': None,
        'best_epoch': None,
        'best_val_ndcg': None,
        'best_val_hr': None,
        'test_ndcg_at_best_val': None,
        'test_hr_at_best_val': None,
    }


def run_spec(python_dir, dataset, seed_index, seed, stage, cfg_name, config, resume, io_retries, retry_backoff_sec):
    config_id = stable_config_id(stage, cfg_name, config)
    train_dir = f'auto_{config_id}_s{seed}'
    run_folder = os.path.join(python_dir, f'{dataset}_{train_dir}')
    log_path = os.path.join(run_folder, 'log.txt')
    legacy_log_path = os.path.join(run_folder, 'test_log.txt')
    existing_log_path = log_path if os.path.isfile(log_path) else legacy_log_path

    if resume and os.path.isfile(existing_log_path):
        metrics = parse_log(existing_log_path)
        if metrics and metrics['status'] == 'ok':
            log_parse_status = metrics.get('status')
            metrics_wo_status = {k: v for k, v in metrics.items() if k != 'status'}
            return {
                'row_type': 'run',
                'status': 'skipped_resume',
                'log_parse_status': log_parse_status,
                'dataset': dataset,
                'repeat': seed_index,
                'seed_index': seed_index,
                'stage': stage,
                'config_name': cfg_name,
                'config_id': config_id,
                'train_dir': train_dir,
                'seed': seed,
                **flatten_config(config),
                **metrics_wo_status,
            }

    cmd = build_command(dataset, train_dir, seed, config)
    os.makedirs(run_folder, exist_ok=True)

    stdout_path = os.path.join(run_folder, 'runner_stdout.log')
    stderr_path = os.path.join(run_folder, 'runner_stderr.log')

    proc = None
    last_exc = None
    for attempt in range(io_retries + 1):
        try:
            with open(stdout_path, 'w', encoding='utf-8') as out, open(stderr_path, 'w', encoding='utf-8') as err:
                proc = subprocess.run(cmd, cwd=python_dir, stdout=out, stderr=err)
            break
        except OSError as exc:
            last_exc = exc
            if attempt < io_retries:
                wait_sec = retry_backoff_sec * (2 ** attempt)
                print(f'[retry] {stage} {cfg_name} seed={seed} OSError({getattr(exc, "errno", "NA")}) while opening log/subprocess I/O, retry {attempt + 1}/{io_retries} after {wait_sec:.2f}s')
                time.sleep(wait_sec)
                continue
            return make_failed_row(
                dataset=dataset,
                seed_index=seed_index,
                seed=seed,
                stage=stage,
                cfg_name=cfg_name,
                config=config,
                config_id=config_id,
                train_dir=train_dir,
                status='failed_io',
                error_message=f'OSError: {last_exc}',
            )
        except Exception as exc:
            return make_failed_row(
                dataset=dataset,
                seed_index=seed_index,
                seed=seed,
                stage=stage,
                cfg_name=cfg_name,
                config=config,
                config_id=config_id,
                train_dir=train_dir,
                status='failed_exception',
                error_message=f'{type(exc).__name__}: {exc}',
            )

    # Prefer the new canonical log filename, but keep compatibility for older runs.
    post_log_path = log_path if os.path.isfile(log_path) else legacy_log_path
    metrics = parse_log(post_log_path)
    if metrics is None:
        metrics = {
            'status': 'missing_log',
            'mean_loss': None,
            'last_loss': None,
            'best_epoch': None,
            'best_val_ndcg': None,
            'best_val_hr': None,
            'test_ndcg_at_best_val': None,
            'test_hr_at_best_val': None,
        }

    status = 'ok' if proc.returncode == 0 and metrics.get('status') == 'ok' else f'failed_rc_{proc.returncode}'
    log_parse_status = metrics.get('status')
    metrics_wo_status = {k: v for k, v in metrics.items() if k != 'status'}

    return {
        'row_type': 'run',
        'status': status,
        'log_parse_status': log_parse_status,
        'dataset': dataset,
        'repeat': seed_index,
        'seed_index': seed_index,
        'stage': stage,
        'config_name': cfg_name,
        'config_id': config_id,
        'train_dir': train_dir,
        'seed': seed,
        **flatten_config(config),
        **metrics_wo_status,
    }


def flatten_config(config):
    out = {}
    for key in sorted(config.keys()):
        if key.endswith('_subset_label'):
            out[key] = config[key]
        else:
            out[key] = bool_to_str(config[key])
    return out


def group_mean(rows, metric):
    values = [r[metric] for r in rows if r.get(metric) is not None]
    if not values:
        return None, None
    return mean(values), (pstdev(values) if len(values) > 1 else 0.0)


def aggregate_rows(run_rows, by_keys):
    buckets = defaultdict(list)
    for row in run_rows:
        if row['status'].startswith('failed') or row['status'] == 'missing_log':
            continue
        key = tuple(row.get(k) for k in by_keys)
        buckets[key].append(row)

    aggs = []
    for key, rows in buckets.items():
        rec = {
            'row_type': 'aggregate',
            'status': 'ok',
            'n_runs': len(rows),
        }
        for idx, k in enumerate(by_keys):
            rec[k] = key[idx]

        for metric in ('best_val_ndcg', 'best_val_hr', 'test_ndcg_at_best_val', 'test_hr_at_best_val', 'mean_loss', 'last_loss'):
            m, s = group_mean(rows, metric)
            rec[f'{metric}_mean'] = m
            rec[f'{metric}_std'] = s

        aggs.append(rec)
    return aggs


def select_best_config(aggregate_rows_list, stage):
    rows = [
        r for r in aggregate_rows_list
        if r.get('stage') == stage
        and r.get('best_val_ndcg_mean') is not None
        and r.get('best_val_hr_mean') is not None
    ]
    if not rows:
        return None

    # Keep selection behavior aligned with main.py spirit: prioritize NDCG,
    # and use HR only as tie-breaker (no additive score).
    return max(
        rows,
        key=lambda r: (
            r['best_val_ndcg_mean'],
            r['best_val_hr_mean'],
        ),
    )


def write_master_table(path, all_rows):
    # Collect all keys so run rows and aggregate rows can coexist in one table
    keys = set()
    for row in all_rows:
        keys.update(row.keys())

    first_keys = [
        'row_type', 'status', 'stage', 'config_name', 'config_id', 'dataset', 'repeat', 'seed_index', 'train_dir', 'seed', 'n_runs'
    ]
    rest_keys = sorted(k for k in keys if k not in first_keys)
    headers = [k for k in first_keys if k in keys] + rest_keys

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Run staged SASRec ablation experiments and collect a master summary table.')
    parser.add_argument('--datasets', nargs='+', default=['ml-1m'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument('--max_parallel', type=int, default=3)
    parser.add_argument('--resume', type=lambda s: s.lower() == 'true', default=True)
    parser.add_argument('--output_csv', default='experiment_master.csv')
    
    parser.add_argument('--io_retries', type=int, default=3)
    parser.add_argument('--retry_backoff_sec', type=float, default=1.0)

    parser.add_argument('--time_ranges', nargs='+', type=int, default=[25, 40, 60])
    parser.add_argument('--time_scales', nargs='+', type=float, default=[1.0, 2.0, 3.0, 4.0])

    parser.add_argument('--short_num_blocks', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--short_kernel_sizes', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--recent_windows', nargs='+', type=int, default=[3, 5])

    parser.add_argument('--gate_hidden_units', nargs='+', type=int, default=[32, 64])

    args = parser.parse_args()
    if args.max_parallel < 1:
        raise ValueError('max_parallel must be >= 1')
    if args.io_retries < 0:
        raise ValueError('io_retries must be >= 0')
    if args.retry_backoff_sec < 0:
        raise ValueError('retry_backoff_sec must be >= 0')

    python_dir = os.path.dirname(os.path.abspath(__file__))

    run_rows = []
    aggregate_all = []

    def run_stage(stage, config_items):
        stage_rows = []
        stage_tasks = []
        for cfg_name, cfg in config_items:
            for dataset in args.datasets:
                total_seeds = len(args.seeds)
                for seed_index, seed in enumerate(args.seeds, start=1):
                    stage_tasks.append((cfg_name, cfg, dataset, seed_index, seed, total_seeds))

        total_tasks = len(stage_tasks)
        max_workers = min(args.max_parallel, total_tasks) if total_tasks > 0 else 1
        print(f'[{stage}] scheduling {total_tasks} runs with max_parallel={max_workers}')

        if total_tasks == 0:
            return [], []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for cfg_name, cfg, dataset, seed_index, seed, _ in stage_tasks:
                fut = executor.submit(
                    run_spec,
                    python_dir,
                    dataset,
                    seed_index,
                    seed,
                    stage,
                    cfg_name,
                    cfg,
                    args.resume,
                    args.io_retries,
                    args.retry_backoff_sec,
                )
                futures[fut] = (cfg_name, cfg, dataset, seed_index, seed)

            finished = 0
            for fut in as_completed(futures):
                cfg_name, cfg, dataset, seed_index, seed = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    config_id = stable_config_id(stage, cfg_name, cfg)
                    train_dir = f'auto_{config_id}_s{seed}'
                    row = make_failed_row(
                        dataset=dataset,
                        seed_index=seed_index,
                        seed=seed,
                        stage=stage,
                        cfg_name=cfg_name,
                        config=cfg,
                        config_id=config_id,
                        train_dir=train_dir,
                        status='failed_future_exception',
                        error_message=f'{type(exc).__name__}: {exc}',
                    )
                stage_rows.append(row)
                finished += 1
                print(f"[{stage}] finished {finished}/{total_tasks} | {row['dataset']} | {row['config_name']} | seed={row['seed']} | status={row['status']}")
        run_rows.extend(stage_rows)

        agg_dataset = aggregate_rows(stage_rows, by_keys=['stage', 'config_name', 'config_id', 'dataset'])
        agg_global = aggregate_rows(stage_rows, by_keys=['stage', 'config_name', 'config_id'])
        for row in agg_dataset:
            row['aggregate_scope'] = 'dataset'
        for row in agg_global:
            row['aggregate_scope'] = 'global'

        aggregate_all.extend(agg_dataset)
        aggregate_all.extend(agg_global)
        return agg_dataset, agg_global

    # Stage A0: approximate original SASRec baseline
    a0_configs = [
        ('baseline_no_time_no_cnn', {
            'use_cnn': False,
            'time_range': 1,
            'time_scale': 0.0,
            'use_normalized_gap': False,
            'use_recent_compactness': False,
            'use_recency_score': False,
            'long_use_normalized_gap': False,
            'long_use_recent_compactness': False,
            'long_use_recency_score': False,
            'cnn_use_normalized_gap': False,
            'cnn_use_recent_compactness': False,
            'cnn_use_recency_score': False,
        })
    ]
    run_stage('A0', a0_configs)

    # Stage A1: discrete bucket sweep, CNN off, continuous features off
    a1_configs = []
    for tr in args.time_ranges:
        for ts in args.time_scales:
            name = f'discrete_tr{tr}_ts{ts}'
            a1_configs.append((name, {
                'use_cnn': False,
                'time_range': tr,
                'time_scale': ts,
                'use_normalized_gap': False,
                'use_recent_compactness': False,
                'use_recency_score': False,
                'long_use_normalized_gap': False,
                'long_use_recent_compactness': False,
                'long_use_recency_score': False,
                'cnn_use_normalized_gap': False,
                'cnn_use_recent_compactness': False,
                'cnn_use_recency_score': False,
            }))
    _, a1_global = run_stage('A1', a1_configs)
    best_a1 = select_best_config(a1_global, 'A1')
    if best_a1 is None:
        raise RuntimeError('No valid configuration found in Stage A1.')

    # Rebuild best A1 params by matching config_name
    best_a1_name = best_a1['config_name']
    best_a1_cfg = dict(next(cfg for name, cfg in a1_configs if name == best_a1_name))

    # Stage A2: long-branch continuous feature subset sweep (CNN off)
    a2_configs = []
    for subset in non_empty_feature_subsets('long'):
        cfg = dict(best_a1_cfg)
        cfg.update({
            'use_cnn': False,
            'long_use_normalized_gap': subset['long_use_normalized_gap'],
            'long_use_recent_compactness': subset['long_use_recent_compactness'],
            'long_use_recency_score': subset['long_use_recency_score'],
            'cnn_use_normalized_gap': False,
            'cnn_use_recent_compactness': False,
            'cnn_use_recency_score': False,
        })
        subset_name = f"long_{subset['long_subset_label']}"
        a2_configs.append((subset_name, cfg))

    _, a2_global = run_stage('A2', a2_configs)
    a2_ranked = sorted(
        [
            r for r in a2_global
            if r.get('stage') == 'A2'
            and r.get('best_val_ndcg_mean') is not None
            and r.get('best_val_hr_mean') is not None
        ],
        key=lambda r: (
            r['best_val_ndcg_mean'],
            r['best_val_hr_mean'],
        ),
        reverse=True,
    )
    if not a2_ranked:
        raise RuntimeError('No valid configuration found in Stage A2.')

    top2_long_names = [r['config_name'] for r in a2_ranked[:2]]
    top1_long_cfg = dict(next(cfg for name, cfg in a2_configs if name == top2_long_names[0]))

    # Stage B1a: CNN structure search with all CNN continuous features enabled
    b1a_configs = []
    for n_blocks, ksz, rw, ghu in itertools.product(
        args.short_num_blocks,
        args.short_kernel_sizes,
        args.recent_windows,
        args.gate_hidden_units,
    ):
        name = f'cnn_struct_b{n_blocks}_k{ksz}_w{rw}_g{ghu}'
        cfg = dict(top1_long_cfg)
        cfg.update({
            'use_cnn': True,
            'short_num_blocks': n_blocks,
            'short_kernel_size': ksz,
            'recent_window': rw,
            'gate_hidden_units': ghu,
            'cnn_use_normalized_gap': True,
            'cnn_use_recent_compactness': True,
            'cnn_use_recency_score': True,
        })
        b1a_configs.append((name, cfg))

    _, b1a_global = run_stage('B1a', b1a_configs)
    best_b1a = select_best_config(b1a_global, 'B1a')
    if best_b1a is None:
        raise RuntimeError('No valid configuration found in Stage B1a.')
    best_b1a_name = best_b1a['config_name']
    best_b1a_cfg = dict(next(cfg for name, cfg in b1a_configs if name == best_b1a_name))

    # Stage B1b: CNN continuous feature subset sweep under best structure
    b1b_configs = []
    for subset in non_empty_feature_subsets('cnn'):
        cfg = dict(best_b1a_cfg)
        cfg.update({
            'cnn_use_normalized_gap': subset['cnn_use_normalized_gap'],
            'cnn_use_recent_compactness': subset['cnn_use_recent_compactness'],
            'cnn_use_recency_score': subset['cnn_use_recency_score'],
        })
        subset_name = f"cnn_{subset['cnn_subset_label']}"
        b1b_configs.append((subset_name, cfg))

    _, b1b_global = run_stage('B1b', b1b_configs)
    b1b_ranked = sorted(
        [
            r for r in b1b_global
            if r.get('stage') == 'B1b'
            and r.get('best_val_ndcg_mean') is not None
            and r.get('best_val_hr_mean') is not None
        ],
        key=lambda r: (
            r['best_val_ndcg_mean'],
            r['best_val_hr_mean'],
        ),
        reverse=True,
    )
    if not b1b_ranked:
        raise RuntimeError('No valid configuration found in Stage B1b.')
    top2_cnn_names = [r['config_name'] for r in b1b_ranked[:2]]

    # Stage B2: top2 long x top2 cnn combination verification
    long_cfg_map = {name: cfg for name, cfg in a2_configs}
    cnn_cfg_map = {name: cfg for name, cfg in b1b_configs}
    b2_configs = []
    for ln, cn in itertools.product(top2_long_names, top2_cnn_names):
        long_cfg = long_cfg_map[ln]
        cnn_cfg = cnn_cfg_map[cn]
        cfg = dict(best_b1a_cfg)
        cfg.update({
            'long_use_normalized_gap': long_cfg['long_use_normalized_gap'],
            'long_use_recent_compactness': long_cfg['long_use_recent_compactness'],
            'long_use_recency_score': long_cfg['long_use_recency_score'],
            'cnn_use_normalized_gap': cnn_cfg['cnn_use_normalized_gap'],
            'cnn_use_recent_compactness': cnn_cfg['cnn_use_recent_compactness'],
            'cnn_use_recency_score': cnn_cfg['cnn_use_recency_score'],
        })
        name = f'long_{ln.split("_", 1)[1]}__cnn_{cn.split("_", 1)[1]}'
        b2_configs.append((name, cfg))

    run_stage('B2', b2_configs)

    master_rows = []
    master_rows.extend(run_rows)
    master_rows.extend(aggregate_all)

    output_path = args.output_csv
    if not os.path.isabs(output_path):
        output_path = os.path.join(python_dir, output_path)
    write_master_table(output_path, master_rows)

    print('\nAll stages finished.')
    print(f'Master table: {output_path}')


if __name__ == '__main__':
    main()
