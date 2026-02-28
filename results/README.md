# 实验结果数据说明文档

## 目录结构

```
results/
├── DeepSeek-V3.2/
│   ├── 1_basic_bandit.jsonl
│   ├── 2_restless_bandit.jsonl
│   ├── 3_contextual_bandit.jsonl
│   ├── 4_adversarial_bandit.jsonl
│   └── 5_sleeping_bandit.jsonl
├── Doubao-Seed-2.0-Code/
├── Doubao-Seed-2.0-lite/
├── Doubao-Seed-2.0-mini/
├── Doubao-Seed-2.0-pro/
├── GLM-4.7/
├── Kimi-K2/
└── progress.json
```

## 数据格式

每行是一个 JSON 对象，包含以下字段：

### 字段说明

| 字段名        | 类型   | 说明                                                   |
| ------------- | ------ | ------------------------------------------------------ |
| `model`       | string | 模型名称，如 "DeepSeek-V3.2"、"Doubao-Seed-2.0-pro" 等 |
| `task`        | string | 任务类型，如 "1_basic_bandit"、"2_restless_bandit" 等  |
| `group`       | int    | 参数组编号（0-9），用于区分不同的实验参数配置          |
| `repeat`      | int    | 重复实验编号（0-4），同一参数组重复 5 次实验           |
| `params`      | object | 实验参数配置                                           |
| `a_reward`    | float  | 策略 A（无解释器）的累计奖励值                         |
| `b_reward`    | float  | 策略 B（带解释器）的累计奖励值                         |
| `improvement` | float  | 策略 B 相对于策略 A 的提升百分比                       |
| `timestamp`   | string | 实验完成时间，格式为 ISO 时间                          |

### params 参数详解

#### 基础老虎机 (basic_bandit)

```json
"params": {
  "n_arms": 3,        // 臂的数量（3-10）
  "mean_low": 3.32,   // 低价值臂的期望奖励下界（2.0-5.0）
  "mean_high": 8.72,  // 高价值臂的期望奖励上界（7.0-9.0）
  "sigma": 1.55       // 噪声标准差（0.5-2.0）
}
```

- 每轮选择一个臂，获得奖励 = 臂的真实期望值 + 高斯噪声
- 目标：最大化累计奖励

#### 静止老虎机 (restless_bandit)

```json
"params": {
  "n_arms": 3,
  "mean_low": 3.32,
  "mean_high": 8.72,
  "sigma": 1.55,
  "drift_rate": 0.05  // 漂移率（0.03-0.08）
}
```

- 与基础老虎机类似，但臂的期望值会随时间缓慢漂移
- drift_rate：每轮期望值的漂移幅度

#### 上下文老虎机 (contextual_bandit)

```json
"params": {
  "n_arms": 3,
  "mean_low": 3.32,
  "mean_high": 8.72,
  "sigma": 1.55,
  "context_dim": 5    // 上下文维度
}
```

- 每个臂有一个上下文向量，上下文会影响臂的期望奖励
- 策略需要根据上下文选择最优臂

#### 对抗性老虎机 (adversarial_bandit)

```json
"params": {
  "n_arms": 3,
  "mean_low": 3.32,
  "mean_high": 8.72,
  "sigma": 1.55,
  "switch_interval": 30  // 切换间隔（20-40）
}
```

- 臂的真实期望值会定期切换（对抗性环境）
- switch_interval：切换发生的周期轮数

#### 休眠老虎机 (sleeping_bandit)

```json
"params": {
  "n_arms": 3,
  "mean_low": 3.32,
  "mean_high": 8.72,
  "sigma": 1.55,
  "sleep_prob": 0.3   // 休眠概率（0.2-0.4）
}
```

- 每轮每个臂有一定概率休眠（不可选）
- 如果所有臂都休眠，则该轮奖励为 0

## 策略说明

### 策略 A (a_reward)

- **无解释器版本**
- 模型直接做决策，不使用代码解释器
- 纯文本推理

### 策略 B (b_reward)

- **带解释器版本**
- 模型使用代码解释器执行代码来辅助决策
- 可以运行计算、验证假设

## improvement 计算公式

```
improvement = (b_reward - a_reward) / abs(a_reward) * 100%
```

- **正值**：策略 B 优于策略 A
- **负值**：策略 B 劣于策略 A
- **0%**：两者相当

## 数据量说明

每个模型应该有的数据量：

- 5 个任务 × 10 个参数组 × 5 次重复 = **250 行**

实际数据量可能因实验中断或 API 错误而少于预期。

## 使用示例

### 读取单个文件

```python
import json

with open('results/DeepSeek-V3.2/1_basic_bandit.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        print(data['model'], data['task'], data['improvement'])
```

### 统计所有模型的平均提升

```python
import json
import os
from collections import defaultdict

results = defaultdict(list)

models = ['DeepSeek-V3.2', 'Doubao-Seed-2.0-Code', 'Doubao-Seed-2.0-lite',
          'Doubao-Seed-2.0-mini', 'Doubao-Seed-2.0-pro', 'GLM-4.7', 'Kimi-K2']

for model in models:
    folder = f'results/{model}'
    for fname in os.listdir(folder):
        if fname.endswith('.jsonl'):
            with open(f'{folder}/{fname}', 'r') as f:
                for line in f:
                    data = json.loads(line)
                    results[model].append(data['improvement'])

for model, improvements in results.items():
    print(f'{model}: {sum(improvements)/len(improvements):.2f}%')
```
