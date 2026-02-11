# strategy_a_no_code/policy.py
import re
import numpy as np
import sys
sys.path.insert(0, '/root/autodl-tmp/ai_study/bandit_study/utils')
from shared import calc_curves

def _parse_action(text, n_arms):
    m = re.search(r"-?\d+", text or "")
    if not m:
        return None
    a = int(m.group())
    if 0 <= a < n_arms:
        return a
    return None

def run_trial_no_code(client, model_id, trial, n_rounds=120, temperature=0.1):
    # 兼容不同类型的trial数据
    means = np.array(trial.get("means", [0] * trial.get("n_arms", 3)), dtype=float)
    reward_table = np.array(trial["rewards"], dtype=float)  # [T, K]
    n_arms = reward_table.shape[1]

    history = {i: [] for i in range(n_arms)}
    actions, rewards = [], []

    for t in range(n_rounds):
        # 只给统计摘要，避免 token 爆炸
        arm_stats = {
            i: {
                "count": len(history[i]),
                "mean": float(np.mean(history[i])) if history[i] else 0.0
            } for i in range(n_arms)
        }

        prompt = (
            f"你在做{n_arms}臂老虎机决策。当前轮次 t={t}。\n"
            f"各臂统计: {arm_stats}\n"
            f"请直接回复下一步动作编号（0到{n_arms-1}之间的整数）。"
            f"不要解释，不要代码。"
        )

        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            raw = resp.choices[0].message.content
            a = _parse_action(raw, n_arms)
            if a is None:
                a = int(np.random.randint(0, n_arms))
        except Exception:
            a = int(np.random.randint(0, n_arms))

        r = float(reward_table[t, a])
        history[a].append(r)
        actions.append(a)
        rewards.append(r)

    # 计算best_mean
    if len(means) > 0 and np.max(means) > 0:
        best_mean = float(np.max(means))
    else:
        best_mean = float(trial.get("best_mean", reward_table.max(axis=1).mean()))
    
    return calc_curves(actions, rewards, best_mean=best_mean)
