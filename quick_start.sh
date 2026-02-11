#!/bin/bash
# Quick Start - 一键运行所有实验并生成可视化

echo "=========================================="
echo "多臂老虎机实验 - Quick Start"
echo "=========================================="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    exit 1
fi

echo "✅ Python环境检查通过"
echo ""

# 步骤1: 运行所有实验
echo "=========================================="
echo "步骤 1/2: 运行所有5类实验"
echo "=========================================="
echo ""
echo "这将运行:"
echo "  - 类1: 基础多臂老虎机"
echo "  - 类2: 非平稳老虎机"
echo "  - 类3: 上下文老虎机"
echo "  - 类4: 对抗性老虎机"
echo "  - 类5: 休眠老虎机"
echo ""
echo "预计耗时: 10-20分钟"
echo ""

python3 run_experiment.py --class all --arms 3 --rounds 120 --trials 10

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 实验运行失败"
    exit 1
fi

echo ""
echo "✅ 所有实验运行完成"
echo ""

# 步骤2: 生成可视化图片
echo "=========================================="
echo "步骤 2/2: 生成可视化图片"
echo "=========================================="
echo ""

python3 generate_plots.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 图片生成失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Quick Start 完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  - 实验数据: experiments/*/results.json"
echo "  - 可视化图: experiments/*/plot.png"
echo "  - 完整文档: docs/飞书文档_五类实验完整版.md"
echo ""
echo "单独运行某个实验:"
echo "  python3 run_experiment.py --class 1  # 只运行类1"
echo "  python3 run_experiment.py --class 2  # 只运行类2"
echo ""

