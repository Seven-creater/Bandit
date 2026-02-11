# bandit_env.py
"""
灵活可配置的多臂老虎机环境
支持多种配置方式和参数自定义
"""
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class BanditConfig:
    """多臂老虎机配置"""
    n_arms: int = 3
    means: Optional[List[float]] = None  # 如果为None，则从分布采样
    mean_low: float = 2.0
    mean_high: float = 9.0
    sigma: float = 1.0  # 噪声标准差
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.means is not None:
            self.n_arms = len(self.means)
            if any(m < 0 for m in self.means):
                raise ValueError("期望值不能为负")


class BanditEnv:
    """
    多臂老虎机环境基类
    支持灵活配置和预生成奖励序列
    """
    
    def __init__(self, config: BanditConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        
        # 初始化真实期望值
        if config.means is not None:
            self.means = np.array(config.means, dtype=float)
        else:
            self.means = self.rng.uniform(
                config.mean_low, 
                config.mean_high, 
                size=config.n_arms
            )
        
        self.n_arms = len(self.means)
        self.best_arm = int(np.argmax(self.means))
        self.best_mean = float(self.means[self.best_arm])
    
    def generate_rewards(self, n_rounds: int) -> np.ndarray:
        """
        预生成 n_rounds 轮的奖励矩阵
        返回: shape=(n_rounds, n_arms) 的奖励矩阵
        """
        noise = self.rng.normal(0, self.config.sigma, size=(n_rounds, self.n_arms))
        rewards = self.means + noise
        return rewards
    
    def get_reward(self, arm: int) -> float:
        """实时采样单次奖励（用于在线交互）"""
        if not 0 <= arm < self.n_arms:
            raise ValueError(f"Invalid arm {arm}, must be in [0, {self.n_arms})")
        return float(self.rng.normal(self.means[arm], self.config.sigma))
    
    def get_info(self) -> Dict[str, Any]:
        """返回环境信息"""
        return {
            "n_arms": self.n_arms,
            "means": self.means.tolist(),
            "best_arm": self.best_arm,
            "best_mean": self.best_mean,
            "sigma": self.config.sigma
        }


def create_trial_configs(
    n_trials: int = 5,
    n_arms_range: tuple = (3, 10),
    mean_low: float = 2.0,
    mean_high: float = 9.0,
    sigma: float = 1.0,
    seed: int = 42
) -> List[BanditConfig]:
    """
    创建多个试验配置，arm数量在指定范围内随机
    
    Args:
        n_trials: 试验数量
        n_arms_range: arm数量范围 (min, max)，包含边界
        mean_low: 期望值下界
        mean_high: 期望值上界
        sigma: 噪声标准差
        seed: 随机种子
    
    Returns:
        配置列表
    """
    rng = np.random.default_rng(seed)
    configs = []
    
    for i in range(n_trials):
        # 随机选择arm数量（3到10之间）
        n_arms = rng.integers(n_arms_range[0], n_arms_range[1] + 1)
        
        config = BanditConfig(
            n_arms=n_arms,
            means=None,  # 让环境自己采样
            mean_low=mean_low,
            mean_high=mean_high,
            sigma=sigma,
            seed=seed + i
        )
        configs.append(config)
    
    return configs


def create_fixed_configs(
    arm_means_list: List[List[float]],
    sigma: float = 1.0
) -> List[BanditConfig]:
    """
    创建固定期望值的配置列表
    
    Args:
        arm_means_list: 每个试验的arm期望值列表
        sigma: 噪声标准差
    
    Returns:
        配置列表
    """
    configs = []
    for i, means in enumerate(arm_means_list):
        config = BanditConfig(
            n_arms=len(means),
            means=means,
            sigma=sigma,
            seed=42 + i
        )
        configs.append(config)
    
    return configs


# 预定义的标准配置
STANDARD_CONFIGS = {
    "easy_3arms": create_fixed_configs([
        [3.0, 5.0, 7.0],  # 差距明显
        [2.5, 5.5, 8.0],
        [3.5, 6.0, 7.5],
    ]),
    
    "hard_3arms": create_fixed_configs([
        [5.0, 5.5, 6.0],  # 差距很小
        [4.8, 5.2, 5.6],
        [5.1, 5.3, 5.5],
    ]),
    
    "mixed_5arms": create_fixed_configs([
        [2.0, 3.5, 5.0, 6.5, 8.0],  # 5个arm，均匀分布
        [3.0, 4.0, 5.0, 6.0, 7.0],
        [2.5, 4.5, 5.5, 6.5, 8.5],
    ]),
    
    "extreme_gap": create_fixed_configs([
        [2.0, 2.5, 9.0],  # 极端差距
        [3.0, 3.5, 8.5],
        [2.5, 3.0, 9.5],
    ]),
    
    # 新增：随机arm数量配置（3-10个arm，5组实验）
    "random_arms": create_trial_configs(
        n_trials=5,
        n_arms_range=(3, 10),
        mean_low=2.0,
        mean_high=9.0,
        sigma=1.0,
        seed=42
    ),
}
