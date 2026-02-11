#!/bin/bash
# 重新运行所有4个实验类别

echo "开始重新运行实验..."

# 类2: 非平稳老虎机
echo "========== 类2: 非平稳老虎机 =========="
cd /root/autodl-tmp/ai_study/bandit_study/experiments/2_restless_bandit
rm -f results.json results_broken.json
/root/autodl-tmp/vllm_env/bin/python run_fixed.py
echo "类2完成！"
echo ""

# 类3: 上下文老虎机
echo "========== 类3: 上下文老虎机 =========="
cd /root/autodl-tmp/ai_study/bandit_study/experiments/3_contextual_bandit
rm -f results.json.backup
/root/autodl-tmp/vllm_env/bin/python run.py
echo "类3完成！"
echo ""

# 类4: 对抗性老虎机
echo "========== 类4: 对抗性老虎机 =========="
cd /root/autodl-tmp/ai_study/bandit_study/experiments/4_adversarial_bandit
rm -f results.json.backup
/root/autodl-tmp/vllm_env/bin/python run.py
echo "类4完成！"
echo ""

# 类5: 休眠老虎机
echo "========== 类5: 休眠老虎机 =========="
cd /root/autodl-tmp/ai_study/bandit_study/experiments/5_sleeping_bandit
rm -f results.json.backup
/root/autodl-tmp/vllm_env/bin/python run.py
echo "类5完成！"
echo ""

echo "✅ 所有实验完成！"

