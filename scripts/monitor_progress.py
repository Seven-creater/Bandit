#!/usr/bin/env python3
"""
监控实验进度并自动生成可视化图片
当检测到新的results.json文件完成时，自动生成对应的plot.png
"""
import os
import json
import time
import subprocess

BASE_DIR = "/root/autodl-tmp/ai_study/bandit_study/experiments"

EXPERIMENTS = {
    "2_restless_bandit": "类2: 非平稳老虎机 (Restless Bandit)",
    "3_contextual_bandit": "类3: 上下文老虎机 (Contextual Bandit)",
    "4_adversarial_bandit": "类4: 对抗性老虎机 (Adversarial Bandit)",
    "5_sleeping_bandit": "类5: 休眠老虎机 (Sleeping Bandit)"
}

def check_and_plot(exp_name, title):
    """检查实验是否完成，如果完成则生成图片"""
    exp_dir = os.path.join(BASE_DIR, exp_name)
    results_path = os.path.join(exp_dir, "results.json")
    plot_path = os.path.join(exp_dir, "plot.png")
    
    # 如果图片已存在，跳过
    if os.path.exists(plot_path):
        return "已完成"
    
    # 检查results.json是否存在且有效
    if not os.path.exists(results_path):
        return "未开始"
    
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        # 检查数据是否完整
        if "A" in data or "A_no_code" in data:
            A = data.get("A", data.get("A_no_code", []))
            B = data.get("B", data.get("B_with_interpreter", []))
            
            if len(A) >= 5 and len(B) >= 5:  # 至少5组实验
                # 调用绘图脚本
                print(f"✅ {exp_name} 数据完整，开始生成图片...")
                cmd = f"cd /root/autodl-tmp/ai_study/bandit_study && python create_plots_345.py"
                subprocess.run(cmd, shell=True, capture_output=True)
                return "生成中"
            else:
                return f"进行中 ({len(A)}/5)"
        else:
            return "数据格式错误"
    except json.JSONDecodeError:
        return "JSON损坏"
    except Exception as e:
        return f"错误: {str(e)}"

def main():
    print("=" * 60)
    print("实验进度监控")
    print("=" * 60)
    
    for exp_name, title in EXPERIMENTS.items():
        status = check_and_plot(exp_name, title)
        print(f"{title:40s} | {status}")
    
    print("=" * 60)
    print("\n提示：")
    print("- 要运行所有实验，请执行: python run_all_experiments.py")
    print("- 要手动生成图片，请执行: python create_all_plots.py")

if __name__ == "__main__":
    main()

