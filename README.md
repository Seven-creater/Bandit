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
├── run_experiment.py                 # 统一实验运行脚本
├── generate_plots.py                 # 可视化生成脚本
└── quick_start.sh                    # 一键运行脚本
```

## 快速开始

### 方式1: 一键运行（推荐）

```bash
# 运行所有实验并生成可视化
./quick_start.sh
```

### 方式2: 分步运行

```bash
# 1. 运行所有实验
python3 run_experiment.py --class all

# 2. 生成可视化图片
python3 generate_plots.py
```

### 方式3: 运行单个实验

```bash
# 只运行类1（基础老虎机）
python3 run_experiment.py --class 1

# 只运行类2（非平稳老虎机）
python3 run_experiment.py --class 2

# 自定义参数
python3 run_experiment.py --class 1 --arms 5 --rounds 200 --trials 20
```

## 查看结果

```bash
# 查看可视化图片
ls experiments/*/plot.png

# 查看实验数据
cat experiments/1_basic_bandit/results.json

# 查看完整研究报告
cat docs/飞书文档_五类实验完整版.md
```

## 实验说明

每类实验包含：
- `results.json` - 实验数据（累积奖励、后悔值、配置等）
- `plot.png` - 可视化图片（4子图对比分析）

### 5类实验

1. **基础多臂老虎机** - 固定奖励分布
2. **非平稳老虎机** - 奖励分布随时间漂移
3. **上下文老虎机** - 带上下文信息的决策
4. **对抗性老虎机** - 奖励分布周期性切换
5. **休眠老虎机** - 臂随机不可用

## 技术栈

- **模型**：Qwen2.5-7B-GPTQ-Int4（vLLM部署）
- **语言**：Python 3.10
- **依赖**：numpy, matplotlib, openai, httpx

## 参数说明

```bash
--class      # 实验类别: 1-5 或 all
--arms       # 臂数量 (默认: 3)
--rounds     # 每个trial的轮数 (默认: 120)
--trials     # trial数量 (默认: 10)
--seed       # 随机种子 (默认: 42)
--verbose    # 显示详细信息
```

## 引用

```
多臂老虎机变体实验：代码解释器增强策略研究
实验时间：2026-02-11
模型：Qwen2.5-7B-GPTQ-Int4
```

## 许可证

MIT License
