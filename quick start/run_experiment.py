#!/usr/bin/env python3
"""
统一实验运行脚本
支持运行单个或所有实验类别
"""
import os
import sys
import json
import argparse
import numpy as np

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)  # 上一级目录是项目根目录
sys.path.insert(0, ROOT)

from utils.shared import get_client_and_model, make_trials
from strategy_a_no_code.policy import run_trial_no_code
from strategy_b_with_interpreter.policy import run_trial_with_interpreter


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def run_basic_bandit(client, model_id, args):
    """类1: 基础多臂老虎机"""
    print("\n" + "="*60)
    print("运行实验类1: 基础多臂老虎机 (Basic MAB)")
    print("="*60)
    
    trials_data = make_trials(
        n_trials=args.trials,
        n_arms=args.arms,
        n_rounds=args.rounds,
        mean_low=2.0,
        mean_high=9.0,
        sigma=1.0,
        seed=args.seed
    )
    
    res_a, res_b = [], []
    for i, tr in enumerate(trials_data, start=1):
        print(f"\nTrial {i}/{args.trials} | means={np.round(tr['means'],2).tolist()}")
        a = run_trial_no_code(client, model_id, tr, n_rounds=args.rounds)
        b = run_trial_with_interpreter(client, model_id, tr, n_rounds=args.rounds, verbose_tool=args.verbose)
        res_a.append(a)
        res_b.append(b)
        print(f"  A累积奖励: {a['cum_reward'][-1]:.2f} | B累积奖励: {b['cum_reward'][-1]:.2f}")
    
    return res_a, res_b


def run_restless_bandit(client, model_id, args):
    """类2: 非平稳老虎机"""
    print("\n" + "="*60)
    print("运行实验类2: 非平稳老虎机 (Restless Bandit)")
    print("="*60)
    
    trials_data = make_trials(
        n_trials=args.trials,
        n_arms=args.arms,
        n_rounds=args.rounds,
        mean_low=2.0,
        mean_high=9.0,
        sigma=1.0,
        seed=args.seed
    )
    
    # 添加漂移参数
    for tr in trials_data:
        tr['drift_rate'] = 0.05
    
    res_a, res_b = [], []
    for i, tr in enumerate(trials_data, start=1):
        print(f"\nTrial {i}/{args.trials} | 初始means={np.round(tr['means'],2).tolist()}")
        a = run_trial_no_code(client, model_id, tr, n_rounds=args.rounds)
        b = run_trial_with_interpreter(client, model_id, tr, n_rounds=args.rounds, verbose_tool=args.verbose)
        res_a.append(a)
        res_b.append(b)
        print(f"  A累积奖励: {a['cum_reward'][-1]:.2f} | B累积奖励: {b['cum_reward'][-1]:.2f}")
    
    return res_a, res_b


def run_contextual_bandit(client, model_id, args):
    """类3: 上下文老虎机"""
    print("\n" + "="*60)
    print("运行实验类3: 上下文老虎机 (Contextual Bandit)")
    print("="*60)
    
    trials_data = make_trials(
        n_trials=args.trials,
        n_arms=args.arms,
        n_rounds=args.rounds,
        mean_low=2.0,
        mean_high=9.0,
        sigma=1.0,
        seed=args.seed
    )
    
    # 添加上下文信息
    for tr in trials_data:
        tr['contexts'] = [f"context_{i%3}" for i in range(args.rounds)]
    
    res_a, res_b = [], []
    for i, tr in enumerate(trials_data, start=1):
        print(f"\nTrial {i}/{args.trials} | means={np.round(tr['means'],2).tolist()}")
        a = run_trial_no_code(client, model_id, tr, n_rounds=args.rounds)
        b = run_trial_with_interpreter(client, model_id, tr, n_rounds=args.rounds, verbose_tool=args.verbose)
        res_a.append(a)
        res_b.append(b)
        print(f"  A累积奖励: {a['cum_reward'][-1]:.2f} | B累积奖励: {b['cum_reward'][-1]:.2f}")
    
    return res_a, res_b


def run_adversarial_bandit(client, model_id, args):
    """类4: 对抗性老虎机"""
    print("\n" + "="*60)
    print("运行实验类4: 对抗性老虎机 (Adversarial Bandit)")
    print("="*60)
    
    trials_data = make_trials(
        n_trials=args.trials,
        n_arms=args.arms,
        n_rounds=args.rounds,
        mean_low=2.0,
        mean_high=9.0,
        sigma=1.0,
        seed=args.seed
    )
    
    # 添加对抗性参数
    for tr in trials_data:
        tr['adversarial'] = True
        tr['switch_interval'] = 30
    
    res_a, res_b = [], []
    for i, tr in enumerate(trials_data, start=1):
        print(f"\nTrial {i}/{args.trials} | 初始means={np.round(tr['means'],2).tolist()}")
        a = run_trial_no_code(client, model_id, tr, n_rounds=args.rounds)
        b = run_trial_with_interpreter(client, model_id, tr, n_rounds=args.rounds, verbose_tool=args.verbose)
        res_a.append(a)
        res_b.append(b)
        print(f"  A累积奖励: {a['cum_reward'][-1]:.2f} | B累积奖励: {b['cum_reward'][-1]:.2f}")
    
    return res_a, res_b


def run_sleeping_bandit(client, model_id, args):
    """类5: 休眠老虎机"""
    print("\n" + "="*60)
    print("运行实验类5: 休眠老虎机 (Sleeping Bandit)")
    print("="*60)
    
    trials_data = make_trials(
        n_trials=args.trials,
        n_arms=args.arms,
        n_rounds=args.rounds,
        mean_low=2.0,
        mean_high=9.0,
        sigma=1.0,
        seed=args.seed
    )
    
    # 添加休眠参数
    for tr in trials_data:
        tr['sleep_prob'] = 0.3
    
    res_a, res_b = [], []
    for i, tr in enumerate(trials_data, start=1):
        print(f"\nTrial {i}/{args.trials} | means={np.round(tr['means'],2).tolist()}")
        a = run_trial_no_code(client, model_id, tr, n_rounds=args.rounds)
        b = run_trial_with_interpreter(client, model_id, tr, n_rounds=args.rounds, verbose_tool=args.verbose)
        res_a.append(a)
        res_b.append(b)
        print(f"  A累积奖励: {a['cum_reward'][-1]:.2f} | B累积奖励: {b['cum_reward'][-1]:.2f}")
    
    return res_a, res_b


def save_results(exp_name, res_a, res_b, args, model_id):
    """保存实验结果"""
    exp_dir = os.path.join(ROOT, "experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    a_final = np.array([x["cum_reward"][-1] for x in res_a], dtype=float)
    b_final = np.array([x["cum_reward"][-1] for x in res_b], dtype=float)
    
    results = {
        "config": {
            "arms": args.arms,
            "rounds": args.rounds,
            "trials": args.trials,
            "seed": args.seed,
            "model": model_id
        },
        "A": res_a,
        "B": res_b,
        "metrics": {
            "A_mean": float(a_final.mean()),
            "A_std": float(a_final.std()),
            "B_mean": float(b_final.mean()),
            "B_std": float(b_final.std()),
            "improvement_pct": float((b_final.mean() - a_final.mean()) / max(1e-9, a_final.mean()) * 100.0)
        }
    }
    
    # 转换numpy类型
    results = convert_numpy_types(results)
    
    output_path = os.path.join(exp_dir, "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存: {output_path}")
    print(f"   策略A平均: {results['metrics']['A_mean']:.2f} ± {results['metrics']['A_std']:.2f}")
    print(f"   策略B平均: {results['metrics']['B_mean']:.2f} ± {results['metrics']['B_std']:.2f}")
    print(f"   提升百分比: {results['metrics']['improvement_pct']:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="运行多臂老虎机实验")
    parser.add_argument("--class", dest="exp_class", type=str, default="all",
                        choices=["1", "2", "3", "4", "5", "all"],
                        help="实验类别: 1-5 或 all (默认: all)")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1",
                        help="API base URL")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key")
    parser.add_argument("--model", type=str, default=None,
                        help="模型ID (可选)")
    parser.add_argument("--arms", type=int, default=3,
                        help="臂数量 (默认: 3)")
    parser.add_argument("--rounds", type=int, default=120,
                        help="每个trial的轮数 (默认: 120)")
    parser.add_argument("--trials", type=int, default=10,
                        help="trial数量 (默认: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细工具调用信息")
    args = parser.parse_args()
    
    # 获取客户端和模型
    client, model_id = get_client_and_model(
        base_url=args.base_url,
        api_key=args.api_key,
        model_override=args.model
    )
    print(f"使用模型: {model_id}")
    
    # 实验映射
    experiments = {
        "1": ("1_basic_bandit", run_basic_bandit),
        "2": ("2_restless_bandit", run_restless_bandit),
        "3": ("3_contextual_bandit", run_contextual_bandit),
        "4": ("4_adversarial_bandit", run_adversarial_bandit),
        "5": ("5_sleeping_bandit", run_sleeping_bandit)
    }
    
    # 确定要运行的实验
    if args.exp_class == "all":
        to_run = list(experiments.keys())
    else:
        to_run = [args.exp_class]
    
    # 运行实验
    for exp_num in to_run:
        exp_name, exp_func = experiments[exp_num]
        res_a, res_b = exp_func(client, model_id, args)
        save_results(exp_name, res_a, res_b, args, model_id)
    
    print("\n" + "="*60)
    print("✅ 所有实验完成！")
    print("="*60)


if __name__ == "__main__":
    main()
