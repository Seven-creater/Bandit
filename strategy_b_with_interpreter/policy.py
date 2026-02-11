# strategy_b_with_interpreter/policy.py
import io
import math
import numpy as np
import contextlib
import sys
sys.path.insert(0, '/root/autodl-tmp/ai_study/bandit_study/utils')
from shared import calc_curves

class PersistentInterpreter:
    def __init__(self, n_arms):
        self.state = {
            "np": np,
            "math": math,
            "n_arms": n_arms,
            "t": 0,
            "history": {i: [] for i in range(n_arms)},
            "choice": 0
        }

    def run(self, code, verbose=False):
        buf = io.StringIO()
        ok, err = True, ""
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, self.state)
        except Exception as e:
            ok, err = False, str(e)

        out = buf.getvalue().strip()
        if verbose:
            print("\n[TOOL CALL] 执行代码:")
            print(code)
            print("[TOOL OUTPUT]")
            print(out if out else "(无输出)")
            if not ok:
                print("[TOOL ERROR]", err)
        return ok, out, err

def fallback_ucb(history, t, n_arms):
    """兜底UCB算法，确保B稳定不崩"""
    if t < n_arms:
        return t
    vals = []
    for i in range(n_arms):
        hist_i = history.get(i, [])
        # 展平列表，防止嵌套
        if hist_i and isinstance(hist_i[0], (list, tuple)):
            flat = []
            for item in hist_i:
                if isinstance(item, (list, tuple)):
                    flat.extend(item)
                else:
                    flat.append(item)
            hist_i = flat
        
        cnt = len(hist_i)
        if cnt == 0:
            vals.append(float("inf"))
            continue
        
        # 确保所有元素都是数值
        hist_i = [float(x) for x in hist_i if isinstance(x, (int, float, np.number))]
        if not hist_i:
            vals.append(float("inf"))
            continue
            
        avg = float(np.mean(hist_i))
        bonus = math.sqrt(2.0 * math.log(t + 1) / cnt)
        vals.append(avg + bonus)
    return int(np.argmax(vals))

def build_policy_code_with_llm(client, model_id, n_arms):
    """
    让 LLM 生成"每轮可执行"的通用 bandit 优化代码（UCB风格）
    """
    prompt = f"""
你需要输出一段 Python 代码（只输出代码，不要解释），用于每轮决策 {n_arms} 臂老虎机。
可用变量：
- t: 当前轮次，从0开始
- n_arms: 臂数量
- history: dict[int, list[float]]，每个臂历史奖励
你必须：
1) 给变量 choice 赋值（0 到 {n_arms-1}）
2) 前 n_arms 轮每个臂至少探索一次
3) 后续使用 UCB1 思路：avg + sqrt(2*log(t+1)/count)
4) print 当前每个臂 count、mean、ucb（用于工具日志）
禁止：
- 不要重置 history
- 不要 import 任何库
"""
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    raw = resp.choices[0].message.content or ""
    # 简单提取代码块
    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            if "python" in p:
                return p.replace("python", "", 1).strip()
        return parts[1].strip() if len(parts) > 1 else raw
    return raw.strip()

def run_trial_with_interpreter(client, model_id, trial, n_rounds=120, verbose_tool=False):
    # 兼容不同类型的trial数据
    means = np.array(trial.get("means", [0] * trial.get("n_arms", 3)), dtype=float)
    reward_table = np.array(trial["rewards"], dtype=float)
    n_arms = reward_table.shape[1]

    interp = PersistentInterpreter(n_arms=n_arms)
    actions, rewards = [], []

    # 只让LLM生成一次策略代码；每轮交给解释器执行（稳定 + 快）
    code = build_policy_code_with_llm(client, model_id, n_arms=n_arms)

    for t in range(n_rounds):
        interp.state["t"] = t
        ok, _, _ = interp.run(code, verbose=verbose_tool and (t < 3 or t % 50 == 0))
        if ok:
            a = interp.state.get("choice", 0)
            try:
                a = int(a)
            except Exception:
                a = fallback_ucb(interp.state["history"], t, n_arms)
        else:
            a = fallback_ucb(interp.state["history"], t, n_arms)

        a = int(np.clip(a, 0, n_arms - 1))
        r = float(reward_table[t, a])

        interp.state["history"][a].append(r)
        actions.append(a)
        rewards.append(r)

    # 计算best_mean
    if len(means) > 0 and np.max(means) > 0:
        best_mean = float(np.max(means))
    else:
        best_mean = float(trial.get("best_mean", reward_table.max(axis=1).mean()))
    
    return calc_curves(actions, rewards, best_mean=best_mean)
