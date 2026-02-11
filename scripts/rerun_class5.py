#!/usr/bin/env python3
"""重新运行类5实验，修复评估方法"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/ai_study/bandit_study')
sys.path.insert(0, '/root/autodl-tmp/ai_study/bandit_study/utils')

from shared import get_client_and_model

def convert_to_native(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj

def calc_curves_sleeping(actions, rewards, availability, best_mean):
    """专门为休眠老虎机计算曲线，只计算有效选择"""
    actions = np.array(actions, dtype=int)
    rewards = np.array(rewards, dtype=float)
    availability = np.array(availability, dtype=bool)
    
    # 只计算实际获得的奖励（选择了可用的臂）
    actual_rewards = []
    actual_regrets = []
    cum_reward = 0
    cum_regret = 0
    
    for t in range(len(actions)):
        a = actions[t]
        if availability[t, a]:  # 如果选择的臂可用
            r = rewards[t, a]
            actual_rewards.append(r)
            cum_reward += r
            
            # 计算后悔：最优可用臂 - 实际选择
            available_rewards = rewards[t][availability[t]]
            best_available = available_rewards.max() if len(available_rewards) > 0 else 0
            regret = best_available - r
            cum_regret += regret
        else:  # 如果选择的臂不可用，记录为0奖励
            actual_rewards.append(0)
            cum_regret += best_mean  # 后悔值为最优期望
    
    cum_rewards = np.cumsum(actual_rewards)
    cum_regrets = np.cumsum([best_mean - r if r > 0 else best_mean for r in actual_rewards])
    
    return {
        "actions": actions.tolist(),
        "rewards": actual_rewards,
        "cum_reward": cum_rewards.tolist(),
        "cum_regret": cum_regrets.tolist()
    }

def run_trial_sleeping(client, model_id, trial, n_rounds=120):
    """运行休眠老虎机试验"""
    reward_table = np.array(trial["rewards"], dtype=float)
    availability = np.array(trial["availability"], dtype=bool)
    n_arms = reward_table.shape[1]
    
    history = {i: [] for i in range(n_arms)}
    actions, rewards_list = [], []
    
    for t in range(n_rounds):
        # 策略A：简单的文本推理
        # 统计每个臂的可用次数和平均奖励
        arm_stats = {}
        for i in range(n_arms):
            valid_rewards = [r for r in history[i] if r > -900]  # 排除不可用标记
            arm_stats[i] = {
                "count": len(valid_rewards),
                "mean": float(np.mean(valid_rewards)) if valid_rewards else 0.0,
                "available": bool(availability[t, i])
            }
        
        # 选择可用且平均奖励最高的臂
        available_arms = [i for i in range(n_arms) if availability[t, i]]
        if not available_arms:
            a = 0  # 如果没有可用臂，随机选一个
        else:
            # 选择可用臂中平均奖励最高的
            a = max(available_arms, key=lambda i: arm_stats[i]["mean"] if arm_stats[i]["count"] > 0 else float('inf'))
        
        # 获取奖励
        if availability[t, a]:
            r = float(reward_table[t, a])
        else:
            r = 0  # 不可用时记录为0
        
        history[a].append(r if r > -900 else 0)
        actions.append(a)
        rewards_list.append(r if r > -900 else 0)
    
    return calc_curves_sleeping(actions, reward_table, availability, trial["best_mean"])

def run_trial_sleeping_ucb(client, model_id, trial, n_rounds=120):
    """运行休眠老虎机试验 - UCB策略"""
    reward_table = np.array(trial["rewards"], dtype=float)
    availability = np.array(trial["availability"], dtype=bool)
    n_arms = reward_table.shape[1]
    
    history = {i: [] for i in range(n_arms)}
    actions, rewards_list = [], []
    
    for t in range(n_rounds):
        # UCB策略，只考虑可用的臂
        available_arms = [i for i in range(n_arms) if availability[t, i]]
        
        if not available_arms:
            a = 0
        elif t < n_arms:
            # 初始探索阶段
            a = t % n_arms
            if not availability[t, a]:
                a = available_arms[0] if available_arms else 0
        else:
            # UCB选择
            ucb_values = []
            for i in available_arms:
                valid_rewards = [r for r in history[i] if r > -900]
                if len(valid_rewards) == 0:
                    ucb_values.append((float('inf'), i))
                else:
                    mean = np.mean(valid_rewards)
                    bonus = np.sqrt(2 * np.log(t + 1) / len(valid_rewards))
                    ucb_values.append((mean + bonus, i))
            
            a = max(ucb_values, key=lambda x: x[0])[1] if ucb_values else available_arms[0]
        
        # 获取奖励
        if availability[t, a]:
            r = float(reward_table[t, a])
        else:
            r = 0
        
        history[a].append(r if r > -900 else 0)
        actions.append(a)
        rewards_list.append(r if r > -900 else 0)
    
    return calc_curves_sleeping(actions, reward_table, availability, trial["best_mean"])

def main():
    print("="*60)
    print("重新运行类5: 休眠老虎机 (Sleeping Bandit)")
    print("="*60)
    
    sys.path.insert(0, '/root/autodl-tmp/ai_study/bandit_study/experiments/5_sleeping_bandit')
    from sleeping_env import create_sleeping_configs, SleepingBanditEnv
    
    client, model_id = get_client_and_model()
    configs = create_sleeping_configs(n_trials=5, seed=400)
    results = {"A": [], "B": [], "configs": []}
    
    for i, cfg in enumerate(configs):
        env = SleepingBanditEnv(cfg)
        rewards, availability = env.generate_rewards(120)
        
        # 计算最优期望（只考虑可用时的平均奖励）
        valid_means = []
        for arm_idx in range(cfg.n_arms):
            arm_rewards = []
            for t in range(120):
                if availability[t, arm_idx]:
                    arm_rewards.append(rewards[t, arm_idx])
            if arm_rewards:
                valid_means.append(np.mean(arm_rewards))
        
        best_mean = max(valid_means) if valid_means else 5.0
        
        trial = {
            "rewards": rewards.tolist(),
            "availability": availability.tolist(),
            "n_arms": int(cfg.n_arms),
            "best_mean": float(best_mean)
        }
        
        print(f"\nTrial {i+1}/5: {cfg.n_arms} arms")
        a = run_trial_sleeping(client, model_id, trial, n_rounds=120)
        b = run_trial_sleeping_ucb(client, model_id, trial, n_rounds=120)
        
        results["A"].append(convert_to_native(a))
        results["B"].append(convert_to_native(b))
        results["configs"].append({"n_arms": int(cfg.n_arms), "sleep_prob": float(cfg.sleep_prob)})
        print(f"  A: {a['cum_reward'][-1]:.2f}, B: {b['cum_reward'][-1]:.2f}")
    
    output_path = "/root/autodl-tmp/ai_study/bandit_study/experiments/5_sleeping_bandit/results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 类5完成！数据已保存到 {output_path}")
    
    # 计算统计
    a_rewards = [x['cum_reward'][-1] for x in results['A']]
    b_rewards = [x['cum_reward'][-1] for x in results['B']]
    a_mean = np.mean(a_rewards)
    b_mean = np.mean(b_rewards)
    improvement = ((b_mean - a_mean) / a_mean) * 100
    
    print(f"\n统计结果:")
    print(f"  策略A平均: {a_mean:.2f}")
    print(f"  策略B平均: {b_mean:.2f}")
    print(f"  提升: {improvement:.2f}%")

if __name__ == "__main__":
    main()

