#!/usr/bin/env python3
"""
最终版本 - 彻底修复所有问题
1. 路径固定在 Bandit/results/
2. 7个模型文件夹 × 5个任务JSONL
3. 实时保存，断点续传
4. 先1%验证，再全量运行
"""
import os
import sys
import json
import yaml
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 固定路径
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(SCRIPT_DIR))

from utils.param_generator import ParamGenerator, create_trial_from_params
from strategy_a_no_code.policy import run_trial_no_code
from strategy_b_with_interpreter.policy import run_trial_with_interpreter

# 全局配置
MAX_WORKERS = 350
API_TIMEOUT = 60
MAX_RETRIES = 3

print_lock = threading.Lock()
progress_lock = threading.Lock()

def safe_print(msg):
    """线程安全打印"""
    with print_lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def append_to_jsonl(filepath, data):
    """追加写入JSONL"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def load_progress(progress_file):
    """加载进度"""
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['completed'] = set(data.get('completed', []))
            data['failed'] = set(data.get('failed', []))
            return data
    return {'completed': set(), 'failed': set(), 'start_time': datetime.now().isoformat()}

def save_progress(progress_file, progress):
    """保存进度"""
    progress_copy = {
        'completed': list(progress['completed']),
        'failed': list(progress['failed']),
        'start_time': progress['start_time'],
        'last_update': datetime.now().isoformat()
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_copy, f, ensure_ascii=False, indent=2)

def make_task_id(model_name, task_id, group_idx):
    """生成任务唯一ID"""
    return f"{model_name}|{task_id}|{group_idx}"

def run_single_param_group(model_info, task_id, params, config, group_idx, progress, progress_file):
    """运行单个参数组（5次重复）"""
    model_name = model_info['name']
    model_id = model_info['model_id']
    exp_cfg = config['experiment']
    
    # 检查是否已完成
    task_uid = make_task_id(model_name, task_id, group_idx)
    with progress_lock:
        if task_uid in progress['completed']:
            return None
    
    # 创建客户端
    client = OpenAI(
        api_key=config['volcengine']['api_key'],
        base_url=config['volcengine']['base_url']
    )
    
    # 模型专属目录和文件
    model_dir = RESULTS_DIR / model_name.replace('/', '_').replace('\\', '_')
    jsonl_file = model_dir / f"{task_id}.jsonl"
    
    success_count = 0
    
    # 5次重复
    for r_idx in range(exp_cfg['n_repeats']):
        trial_params = params.copy()
        trial_params['seed'] = params['seed'] + r_idx
        trial = create_trial_from_params(trial_params, exp_cfg['n_rounds'])
        
        for retry in range(MAX_RETRIES):
            try:
                # 运行策略A和B
                result_a = run_trial_no_code(client, model_id, trial, n_rounds=exp_cfg['n_rounds'])
                result_b = run_trial_with_interpreter(client, model_id, trial, n_rounds=exp_cfg['n_rounds'], verbose_tool=False)
                
                # 保存结果
                result_data = {
                    'model': model_name,
                    'task': task_id,
                    'group': group_idx,
                    'repeat': r_idx,
                    'params': {
                        'n_arms': params['n_arms'],
                        'mean_low': round(params['mean_low'], 2),
                        'mean_high': round(params['mean_high'], 2),
                        'sigma': round(params['sigma'], 2)
                    },
                    'a_reward': float(result_a['cum_reward'][-1]),
                    'b_reward': float(result_b['cum_reward'][-1]),
                    'improvement': float((result_b['cum_reward'][-1] - result_a['cum_reward'][-1]) / max(result_a['cum_reward'][-1], 1e-9) * 100),
                    'timestamp': datetime.now().isoformat()
                }
                
                # 立即追加到JSONL
                append_to_jsonl(jsonl_file, result_data)
                success_count += 1
                break
                
            except Exception as e:
                if retry < MAX_RETRIES - 1:
                    time.sleep(2)
                else:
                    # 记录失败
                    failed_data = {
                        'model': model_name,
                        'task': task_id,
                        'group': group_idx,
                        'repeat': r_idx,
                        'error': str(e)[:200],
                        'timestamp': datetime.now().isoformat()
                    }
                    failed_file = RESULTS_DIR / "failed.jsonl"
                    append_to_jsonl(failed_file, failed_data)
    
    # 标记完成
    with progress_lock:
        progress['completed'].add(task_uid)
        save_progress(progress_file, progress)
    
    safe_print(f"[完成] {model_name} | {task_id} | 组{group_idx+1} ({success_count}/5成功)")
    
    return task_uid

def validate_setup(config):
    """阶段1: 快速验证"""
    safe_print("\n" + "="*70)
    safe_print("阶段1: 快速验证（1%数据，预计1-2分钟）")
    safe_print("="*70)
    
    models = [m for m in config['models'] if m.get('enabled', True)]
    test_model = models[0]
    
    safe_print(f"测试模型: {test_model['name']}")
    safe_print(f"结果目录: {RESULTS_DIR}")
    
    client = OpenAI(
        api_key=config['volcengine']['api_key'],
        base_url=config['volcengine']['base_url']
    )
    
    test_params = {
        'n_arms': 3,
        'mean_low': 3.0,
        'mean_high': 8.0,
        'sigma': 1.0,
        'seed': 42
    }
    
    trial = create_trial_from_params(test_params, n_rounds=30)
    
    try:
        safe_print("测试策略A...")
        result_a = run_trial_no_code(client, test_model['model_id'], trial, n_rounds=30)
        safe_print(f"✅ 策略A成功: 累积奖励={result_a['cum_reward'][-1]:.1f}")
        
        safe_print("测试策略B...")
        result_b = run_trial_with_interpreter(client, test_model['model_id'], trial, n_rounds=30, verbose_tool=False)
        safe_print(f"✅ 策略B成功: 累积奖励={result_b['cum_reward'][-1]:.1f}")
        
        # 测试文件写入
        test_dir = RESULTS_DIR / test_model['name'].replace('/', '_').replace('\\', '_')
        test_file = test_dir / "test.jsonl"
        append_to_jsonl(test_file, {'test': 'success', 'timestamp': datetime.now().isoformat()})
        
        if test_file.exists():
            safe_print(f"✅ 文件写入成功: {test_file}")
            safe_print(f"✅ 文件大小: {test_file.stat().st_size} bytes")
        else:
            raise Exception("文件写入失败")
        
        safe_print("\n✅ 验证通过！可以开始全量运行")
        return True
        
    except Exception as e:
        safe_print(f"\n❌ 验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_full_experiment(config):
    """阶段2: 全量运行"""
    start_time = time.time()
    
    safe_print("\n" + "="*70)
    safe_print("阶段2: 全量运行（350并发，预计30-60分钟）")
    safe_print("="*70)
    
    exp_cfg = config['experiment']
    models = [m for m in config['models'] if m.get('enabled', True)]
    
    safe_print(f"模型数量: {len(models)}")
    safe_print(f"任务数量: 5")
    safe_print(f"参数组数: {exp_cfg['n_param_groups']}")
    safe_print(f"重复次数: {exp_cfg['n_repeats']}")
    safe_print(f"最大并发: {MAX_WORKERS}")
    safe_print(f"结果目录: {RESULTS_DIR}")
    safe_print(f"文件结构: {len(models)}个模型文件夹 × 5个任务JSONL")
    safe_print("="*70 + "\n")
    
    # 生成参数
    param_gen = ParamGenerator(exp_cfg, seed=exp_cfg['seed'])
    all_params = param_gen.generate_all_params(exp_cfg['n_param_groups'])
    
    # 任务列表
    tasks = [
        ('1_basic_bandit', 'basic'),
        ('2_restless_bandit', 'restless'),
        ('3_contextual_bandit', 'contextual'),
        ('4_adversarial_bandit', 'adversarial'),
        ('5_sleeping_bandit', 'sleeping')
    ]
    
    # 创建输出目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载进度
    progress_file = RESULTS_DIR / "progress.json"
    progress = load_progress(progress_file)
    
    # 生成所有任务
    all_jobs = []
    for task_id, task_key in tasks:
        params_list = all_params[task_key]
        for model_info in models:
            for group_idx, params in enumerate(params_list):
                task_uid = make_task_id(model_info['name'], task_id, group_idx)
                if task_uid not in progress['completed']:
                    all_jobs.append({
                        'model_info': model_info,
                        'task_id': task_id,
                        'params': params,
                        'group_idx': group_idx
                    })
    
    total_jobs = len(all_jobs)
    safe_print(f"总任务数: {total_jobs} (已完成: {len(progress['completed'])})")
    
    if total_jobs == 0:
        safe_print("所有任务已完成！")
        generate_final_report(config, models, tasks, start_time)
        return
    
    safe_print("开始并发执行...\n")
    
    # 并发执行
    completed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                run_single_param_group,
                job['model_info'],
                job['task_id'],
                job['params'],
                config,
                job['group_idx'],
                progress,
                progress_file
            ): job
            for job in all_jobs
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    completed_count += 1
                    percentage = (completed_count / total_jobs * 100)
                    safe_print(f"[总进度] {completed_count}/{total_jobs} ({percentage:.1f}%)")
                    
            except Exception as e:
                safe_print(f"[错误] {str(e)[:100]}")
    
    # 生成最终报告
    generate_final_report(config, models, tasks, start_time)

def generate_final_report(config, models, tasks, start_time):
    """生成最终报告"""
    safe_print("\n" + "="*70)
    safe_print("生成最终报告...")
    safe_print("="*70)
    
    report_lines = []
    report_lines.append("# 7模型并发Bandit实验报告\n\n")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    elapsed = time.time() - start_time
    report_lines.append(f"## 实验信息\n\n")
    report_lines.append(f"- 运行时间: {elapsed/60:.1f} 分钟\n")
    report_lines.append(f"- 模型数量: {len(models)}\n")
    report_lines.append(f"- 结果目录: {RESULTS_DIR}\n\n")
    
    report_lines.append("## 测试模型\n\n")
    for m in models:
        report_lines.append(f"- {m['name']}\n")
    report_lines.append("\n")
    
    report_lines.append("## 实验结果\n\n")
    
    for task_id, task_key in tasks:
        report_lines.append(f"### {task_id}\n\n")
        report_lines.append("| 模型 | 策略A | 策略B | 提升(%) | 实验次数 |\n")
        report_lines.append("|------|-------|-------|---------|----------|\n")
        
        for model_info in models:
            model_name = model_info['name']
            model_dir = RESULTS_DIR / model_name.replace('/', '_').replace('\\', '_')
            jsonl_file = model_dir / f"{task_id}.jsonl"
            
            if jsonl_file.exists():
                a_vals, b_vals = [], []
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        a_vals.append(data['a_reward'])
                        b_vals.append(data['b_reward'])
                
                if a_vals and b_vals:
                    a_mean = np.mean(a_vals)
                    a_std = np.std(a_vals)
                    b_mean = np.mean(b_vals)
                    b_std = np.std(b_vals)
                    improvement = (b_mean - a_mean) / max(a_mean, 1e-9) * 100
                    
                    report_lines.append(f"| {model_name} | {a_mean:.1f}±{a_std:.1f} | "
                                      f"{b_mean:.1f}±{b_std:.1f} | "
                                      f"{improvement:.1f}% | {len(a_vals)} |\n")
        
        report_lines.append("\n")
    
    # 写入报告
    report_file = RESULTS_DIR / "final_summary.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    safe_print(f"✅ 报告已生成: {report_file}")

def main():
    """主函数"""
    safe_print(f"脚本目录: {SCRIPT_DIR}")
    safe_print(f"结果目录: {RESULTS_DIR}")
    
    config_file = SCRIPT_DIR / 'config.yaml'
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 阶段1: 快速验证
    if not validate_setup(config):
        safe_print("\n❌ 验证失败，终止运行")
        return
    
    # 阶段2: 全量运行
    run_full_experiment(config)
    
    safe_print("\n" + "="*70)
    safe_print("✅ 所有实验完成！")
    safe_print(f"结果保存在: {RESULTS_DIR}")
    safe_print("="*70)

if __name__ == "__main__":
    main()

