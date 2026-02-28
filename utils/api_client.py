"""
统一API客户端
支持火山云等兼容OpenAI格式的API
"""
import yaml
from openai import OpenAI
from typing import Optional, Dict, Any
import os


class UnifiedAPIClient:
    """统一的API客户端，从配置文件读取"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化API客户端
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.base_url = self.config['volcengine']['base_url']
        self.api_key = self.config['volcengine']['api_key']
        
        # 创建OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_enabled_models(self):
        """获取所有启用的模型"""
        return [
            model for model in self.config['models']
            if model.get('enabled', True)
        ]
    
    def get_client(self) -> OpenAI:
        """获取OpenAI客户端实例"""
        return self.client
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """获取实验配置"""
        return self.config.get('experiment', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """获取输出配置"""
        return self.config.get('output', {})
    
    def chat_completion(self, model_id: str, messages: list, temperature: float = 0.1, **kwargs):
        """
        调用聊天补全API
        
        Args:
            model_id: 模型ID（endpoint ID）
            messages: 消息列表
            temperature: 温度参数
            **kwargs: 其他参数
        
        Returns:
            API响应
        """
        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
            return response
        except Exception as e:
            print(f"API调用失败: {e}")
            raise


def get_api_client(config_path: str = "config.yaml") -> UnifiedAPIClient:
    """
    获取API客户端实例（工厂函数）
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        UnifiedAPIClient实例
    """
    return UnifiedAPIClient(config_path)

