# shared.py
import numpy as np
from openai import OpenAI
from typing import List, Dict, Any
import os
import sys

# 添加当前目录到路径以支持相对导入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from utils.bandit_env import BanditEnv, BanditConfig
except ImportError:
    from bandit_env import BanditEnv, BanditConfig

def get_client_and_model(base_url="http://localhost:8000/v1", api_key="EMPTY", model_override=None):
    client = OpenAI(api_key=api_key, base_url=base_url)
    if model_override:
        return client, model_override
    model_id = client.models.list().data[0].id
    return client, model_id

def make_trials_from_configs(configs: List[BanditConfig], n_rounds: int = 120) -> List[Dict[str, Any]]:
    """
    从配置列表生成试验数据
    
    Args:
        configs: BanditConfig 列表
        n_rounds: 每个试验的轮数
    
    Returns:
        trial 列表，每个包含 means 和 rewards
    """
    trials = []
    for config in configs:
        env = BanditEnv(config)
        rewards = env.generate_rewards(n_rounds)
        trials.append({
            "means": env.means.tolist(),
            "rewards": rewards.tolist(),
            "n_arms": env.n_arms,
            "best_arm": env.best_arm,
            "best_mean": env.best_mean,
            "config": {
                "sigma": config.sigma,
                "mean_low": config.mean_low,
                "mean_high": config.mean_high,
            }
        })
    return trials

def make_trials(n_trials=10, n_arms=3, n_rounds=120, mean_low=2.0, mean_high=9.0, sigma=1.0, seed=42):
    """
    预生成试验数据，确保 A/B 在同一 trial 上完全公平比较
    (保留旧接口以兼容现有代码)
    """
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_trials):
        means = rng.uniform(mean_low, mean_high, size=n_arms)
        noise = rng.normal(0, sigma, size=(n_rounds, n_arms))
        rewards = means + noise
        trials.append({
            "means": means.tolist(),
            "rewards": rewards.tolist()
        })
    return trials

def calc_curves(actions, rewards, best_mean):
    rewards = np.array(rewards, dtype=float)
    cum_reward = np.cumsum(rewards)
    regret = best_mean - rewards
    cum_regret = np.cumsum(regret)
    return {
        "actions": [int(x) for x in actions],
        "rewards": rewards.tolist(),
        "cum_reward": cum_reward.tolist(),
        "cum_regret": cum_regret.tolist()
    }
