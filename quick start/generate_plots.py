#!/usr/bin/env python3
"""
生成所有实验的可视化图片
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)  # 上一级目录是项目根目录


def plot_experiment(exp_dir, title):
    """为单个实验生成可视化图片"""
    results_path = os.path.join(exp_dir, "results.json")
    
    if not os.path.exists(results_path):
        print(f"⚠️  跳过 {os.path.basename(exp_dir)}: results.json 不存在")
        return False
    
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️  跳过 {os.path.basename(exp_dir)}: JSON解析失败 - {e}")
        return False
    
    # 提取数据
    A = data.get("A", data.get("A_no_code", []))
    B = data.get("B", data.get("B_with_interpreter", []))
    
    if len(A) == 0 or len(B) == 0:
        print(f"⚠️  跳过 {os.path.basename(exp_dir)}: 数据为空")
        return False
    
    # 提取累积奖励和后悔
    A_reward = np.array([x["cum_reward"] for x in A], dtype=float)
    B_reward = np.array([x["cum_reward"] for x in B], dtype=float)
    A_regret = np.array([x["cum_regret"] for x in A], dtype=float)
    B_regret = np.array([x["cum_regret"] for x in B], dtype=float)
    
    t = np.arange(A_reward.shape[1])
    
    a_mr, a_sr = A_reward.mean(axis=0), A_reward.std(axis=0)
    b_mr, b_sr = B_reward.mean(axis=0), B_reward.std(axis=0)
    a_mg, a_sg = A_regret.mean(axis=0), A_regret.std(axis=0)
    b_mg, b_sg = B_regret.mean(axis=0), B_regret.std(axis=0)
    
    # 创建2x2子图
    plt.figure(figsize=(14, 10))
    
    # 1. 累积奖励
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, a_mr, 'r--', linewidth=2, label='Strategy A: No Code LLM')
    ax1.fill_between(t, a_mr-a_sr, a_mr+a_sr, color='r', alpha=0.2)
    ax1.plot(t, b_mr, 'g-', linewidth=2, label='Strategy B: Interpreter+UCB')
    ax1.fill_between(t, b_mr-b_sr, b_mr+b_sr, color='g', alpha=0.2)
    ax1.set_title("Cumulative Reward (Higher Better)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time Steps", fontsize=11)
    ax1.set_ylabel("Cumulative Reward", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. 累积后悔
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t, a_mg, 'r--', linewidth=2, label='Strategy A')
    ax2.fill_between(t, a_mg-a_sg, a_mg+a_sg, color='r', alpha=0.2)
    ax2.plot(t, b_mg, 'g-', linewidth=2, label='Strategy B')
    ax2.fill_between(t, b_mg-b_sg, b_mg+b_sg, color='g', alpha=0.2)
    ax2.set_title("Cumulative Regret (Lower Better)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time Steps", fontsize=11)
    ax2.set_ylabel("Cumulative Regret", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # 3. 平均每步奖励
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(t, a_mr/(t+1), 'r--', linewidth=2, label='Strategy A Avg')
    ax3.plot(t, b_mr/(t+1), 'g-', linewidth=2, label='Strategy B Avg')
    ax3.set_title("Average Reward Per Step", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Time Steps", fontsize=11)
    ax3.set_ylabel("Average Reward", fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 4. 最终累积奖励对比
    ax4 = plt.subplot(2, 2, 4)
    a_final = A_reward[:, -1]
    b_final = B_reward[:, -1]
    improvement = ((b_final.mean() - a_final.mean()) / a_final.mean()) * 100
    
    bars = ax4.bar(["Strategy A\nNo Code", "Strategy B\nInterpreter"], 
                   [a_final.mean(), b_final.mean()],
                   yerr=[a_final.std(), b_final.std()], 
                   color=["#ff6666", "#66cc66"],
                   capsize=8, width=0.6)
    ax4.set_title(f"Final Cumulative Reward (Improvement: {improvement:.1f}%)", 
                  fontsize=12, fontweight='bold')
    ax4.set_ylabel("Cumulative Reward", fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    
    # 在柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(exp_dir, "plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 已生成: {output_path}")
    plt.close()
    return True


def main():
    experiments_dir = os.path.join(ROOT, "experiments")
    
    experiments = [
        ("1_basic_bandit", "Class 1: Basic Multi-Armed Bandit"),
        ("2_restless_bandit", "Class 2: Restless Bandit"),
        ("3_contextual_bandit", "Class 3: Contextual Bandit"),
        ("4_adversarial_bandit", "Class 4: Adversarial Bandit"),
        ("5_sleeping_bandit", "Class 5: Sleeping Bandit")
    ]
    
    print("="*60)
    print("开始生成所有可视化图片")
    print("="*60 + "\n")
    
    success_count = 0
    for exp_name, title in experiments:
        exp_dir = os.path.join(experiments_dir, exp_name)
        if plot_experiment(exp_dir, title):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ 成功生成 {success_count}/{len(experiments)} 个图片")
    print("="*60)


if __name__ == "__main__":
    main()
