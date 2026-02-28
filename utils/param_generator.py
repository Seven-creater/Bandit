"""
参数生成器
为每类Bandit任务生成随机参数组
"""
import numpy as np
from typing import List, Dict, Any, Tuple


class ParamGenerator:
    """Bandit参数生成器"""
    
    def __init__(self, config: Dict[str, Any], seed: int = 42):
        """
        初始化参数生成器
        
        Args:
            config: 实验配置字典
            seed: 随机种子
        """
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def generate_basic_params(self, n_groups: int = 10) -> List[Dict[str, Any]]:
        """
        生成基础Bandit参数
        
        Args:
            n_groups: 参数组数
            
        Returns:
            参数字典列表
        """
        params_list = []
        bandit_cfg = self.config.get('bandit_params', {})
        
        n_arms_range = bandit_cfg.get('n_arms_range', [3, 10])
        mean_low_range = bandit_cfg.get('mean_low_range', [2.0, 5.0])
        mean_high_range = bandit_cfg.get('mean_high_range', [7.0, 9.0])
        sigma_range = bandit_cfg.get('sigma_range', [0.5, 2.0])
        
        for i in range(n_groups):
            params = {
                'n_arms': int(self.rng.integers(n_arms_range[0], n_arms_range[1] + 1)),
                'mean_low': float(self.rng.uniform(mean_low_range[0], mean_low_range[1])),
                'mean_high': float(self.rng.uniform(mean_high_range[0], mean_high_range[1])),
                'sigma': float(self.rng.uniform(sigma_range[0], sigma_range[1])),
                'seed': self.seed + i * 100,
                'group_id': i
            }
            params_list.append(params)
        
        return params_list
    
    def generate_restless_params(self, n_groups: int = 10) -> List[Dict[str, Any]]:
        """生成非平稳Bandit参数"""
        params_list = self.generate_basic_params(n_groups)
        
        restless_cfg = self.config.get('restless_params', {})
        drift_range = restless_cfg.get('drift_rate_range', [0.03, 0.08])
        
        for params in params_list:
            params['drift_rate'] = float(self.rng.uniform(drift_range[0], drift_range[1]))
        
        return params_list
    
    def generate_contextual_params(self, n_groups: int = 10) -> List[Dict[str, Any]]:
        """生成上下文Bandit参数"""
        params_list = self.generate_basic_params(n_groups)
        
        for params in params_list:
            # 上下文数量在2-5之间
            params['n_contexts'] = int(self.rng.integers(2, 6))
        
        return params_list
    
    def generate_adversarial_params(self, n_groups: int = 10) -> List[Dict[str, Any]]:
        """生成对抗性Bandit参数"""
        params_list = self.generate_basic_params(n_groups)
        
        adv_cfg = self.config.get('adversarial_params', {})
        switch_range = adv_cfg.get('switch_interval_range', [20, 40])
        
        for params in params_list:
            params['adversarial'] = True
            params['switch_interval'] = int(self.rng.integers(switch_range[0], switch_range[1] + 1))
        
        return params_list
    
    def generate_sleeping_params(self, n_groups: int = 10) -> List[Dict[str, Any]]:
        """生成休眠Bandit参数"""
        params_list = self.generate_basic_params(n_groups)
        
        sleep_cfg = self.config.get('sleeping_params', {})
        sleep_range = sleep_cfg.get('sleep_prob_range', [0.2, 0.4])
        
        for params in params_list:
            params['sleep_prob'] = float(self.rng.uniform(sleep_range[0], sleep_range[1]))
        
        return params_list
    
    def generate_all_params(self, n_groups: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        生成所有5类任务的参数
        
        Args:
            n_groups: 每类任务的参数组数
            
        Returns:
            字典，键为任务类型，值为参数列表
        """
        return {
            'basic': self.generate_basic_params(n_groups),
            'restless': self.generate_restless_params(n_groups),
            'contextual': self.generate_contextual_params(n_groups),
            'adversarial': self.generate_adversarial_params(n_groups),
            'sleeping': self.generate_sleeping_params(n_groups)
        }


def create_trial_from_params(params: Dict[str, Any], n_rounds: int = 120) -> Dict[str, Any]:
    """
    从参数字典创建trial数据
    
    Args:
        params: 参数字典
        n_rounds: 轮数
        
    Returns:
        trial字典，包含means和rewards
    """
    rng = np.random.default_rng(params['seed'])
    
    # 生成期望值
    means = rng.uniform(
        params['mean_low'],
        params['mean_high'],
        size=params['n_arms']
    )
    
    # 生成奖励矩阵
    noise = rng.normal(0, params['sigma'], size=(n_rounds, params['n_arms']))
    
    # 处理非平稳情况
    if 'drift_rate' in params:
        drift = np.cumsum(
            rng.normal(0, params['drift_rate'], size=(n_rounds, params['n_arms'])),
            axis=0
        )
        rewards = means + noise + drift
    else:
        rewards = means + noise
    
    trial = {
        'means': means.tolist(),
        'rewards': rewards.tolist(),
        'n_arms': params['n_arms'],
        'best_arm': int(np.argmax(means)),
        'best_mean': float(np.max(means)),
        'params': params
    }
    
    # 添加特定任务的额外信息
    if 'drift_rate' in params:
        trial['drift_rate'] = params['drift_rate']
    
    if 'n_contexts' in params:
        trial['contexts'] = [f"context_{i % params['n_contexts']}" for i in range(n_rounds)]
    
    if 'adversarial' in params:
        trial['adversarial'] = params['adversarial']
        trial['switch_interval'] = params['switch_interval']
    
    if 'sleep_prob' in params:
        trial['sleep_prob'] = params['sleep_prob']
    
    return trial

