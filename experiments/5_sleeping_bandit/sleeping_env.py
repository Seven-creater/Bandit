# sleeping_env.py - 休眠老虎机
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class SleepingConfig:
    n_arms: int = 5
    mean_low: float = 2.0
    mean_high: float = 9.0
    sigma: float = 1.0
    sleep_prob: float = 0.3
    seed: Optional[int] = None

class SleepingBanditEnv:
    def __init__(self, config: SleepingConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.means = self.rng.uniform(config.mean_low, config.mean_high, size=config.n_arms)
        self.n_arms = config.n_arms
    
    def generate_rewards(self, n_rounds: int):
        rewards = np.zeros((n_rounds, self.n_arms))
        availability = np.zeros((n_rounds, self.n_arms), dtype=bool)
        
        for t in range(n_rounds):
            availability[t] = self.rng.random(self.n_arms) > self.config.sleep_prob
            if not availability[t].any():
                availability[t, self.rng.integers(0, self.n_arms)] = True
            
            for i in range(self.n_arms):
                if availability[t, i]:
                    rewards[t, i] = self.rng.normal(self.means[i], self.config.sigma)
                else:
                    rewards[t, i] = -999
        
        return rewards, availability

def create_sleeping_configs(n_trials=5, seed=42):
    rng = np.random.default_rng(seed)
    configs = []
    for i in range(n_trials):
        n_arms = rng.integers(3, 11)
        config = SleepingConfig(n_arms=n_arms, sleep_prob=0.3, seed=seed+i)
        configs.append(config)
    return configs

