# 多臂老虎机变体实验项目

本项目对比两种决策策略在五类多臂老虎机（Multi-Armed Bandit, MAB）变体上的表现。

## 核心发现

✅ **策略B（代码解释器+UCB）在5类实验中平均提升26.39%**

- 类1（基础）：+88.45%
- 类2（非平稳）：+37.80%
- 类3（上下文）：+3.03%
- 类4（对抗性）：+37.85%
- 类5（休眠）：-0.76%（持平）

## 项目结构

```
bandit_study/
├── docs/                              # 文档
│   └── 飞书文档_五类实验完整版.md      # 完整研究报告
├── experiments/                       # 实验数据（5类）
│   ├── 1_basic_bandit/
│   ├── 2_restless_bandit/
│   ├── 3_contextual_bandit/
│   ├── 4_adversarial_bandit/
│   └── 5_sleeping_bandit/
├── strategy_a_no_code/               # 策略A：无代码LLM推理
├── strategy_b_with_interpreter/      # 策略B：代码解释器+UCB
├── utils/                            # 工具模块
└── scripts/                          # 辅助脚本
```

## 快速开始

### 查看实验结果

```bash
# 查看可视化图片
ls experiments/*/plot.png

# 查看实验数据
cat experiments/1_basic_bandit/results.json

# 查看完整研究报告
cat docs/飞书文档_五类实验完整版.md
```

## 实验结果

所有5类实验已完成，每类包含：
- `results.json` - 实验数据
- `plot.png` - 可视化图片（4子图对比分析）

## 技术栈

- **模型**：Qwen2.5-7B-GPTQ-Int4（vLLM部署）
- **语言**：Python 3.10
- **依赖**：numpy, matplotlib, openai, httpx

## 引用

```
多臂老虎机变体实验：代码解释器增强策略研究
实验时间：2026-02-11
模型：Qwen2.5-7B-GPTQ-Int4
```

## 许可证

MIT License

