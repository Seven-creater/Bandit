# run_compare.py
import os
import json
import argparse
import numpy as np
import sys

# 保证从当前项目根目录导入
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared import get_client_and_model, make_trials
from strategy_a_no_code.policy import run_trial_no_code
from strategy_b_with_interpreter.policy import run_trial_with_interpreter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default=None, help="可选，手动指定模型ID")
    parser.add_argument("--arms", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--mean_low", type=float, default=2.0)
    parser.add_argument("--mean_high", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose_tool", action="store_true")
    args = parser.parse_args()

    client, model_id = get_client_and_model(
        base_url=args.base_url,
        api_key=args.api_key,
        model_override=args.model
    )
    print(f"[INFO] Using model: {model_id}")

    trials_data = make_trials(
        n_trials=args.trials,
        n_arms=args.arms,
        n_rounds=args.rounds,
        mean_low=args.mean_low,
        mean_high=args.mean_high,
        sigma=args.sigma,
        seed=args.seed
    )

    res_a, res_b = [], []
    for i, tr in enumerate(trials_data, start=1):
        print(f"\n=== Trial {i}/{args.trials} | means={np.round(tr['means'],2).tolist()} ===")

        a = run_trial_no_code(client, model_id, tr, n_rounds=args.rounds)
        b = run_trial_with_interpreter(client, model_id, tr, n_rounds=args.rounds, verbose_tool=args.verbose_tool)

        res_a.append(a)
        res_b.append(b)

        print(f"A final reward: {a['cum_reward'][-1]:.2f}")
        print(f"B final reward: {b['cum_reward'][-1]:.2f}")

    # 汇总
    a_final = np.array([x["cum_reward"][-1] for x in res_a], dtype=float)
    b_final = np.array([x["cum_reward"][-1] for x in res_b], dtype=float)

    summary = {
        "config": vars(args),
        "model_id": model_id,
        "A_no_code": res_a,
        "B_with_interpreter": res_b,
        "metrics": {
            "A_mean_final_reward": float(a_final.mean()),
            "A_std_final_reward": float(a_final.std()),
            "B_mean_final_reward": float(b_final.mean()),
            "B_std_final_reward": float(b_final.std()),
            "improve_pct": float((b_final.mean() - a_final.mean()) / max(1e-9, a_final.mean()) * 100.0)
        }
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/compare_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== SUMMARY =====")
    print(f"A mean final reward: {summary['metrics']['A_mean_final_reward']:.2f} ± {summary['metrics']['A_std_final_reward']:.2f}")
    print(f"B mean final reward: {summary['metrics']['B_mean_final_reward']:.2f} ± {summary['metrics']['B_std_final_reward']:.2f}")
    print(f"Improvement % (B vs A): {summary['metrics']['improve_pct']:.2f}%")
    print("Saved: outputs/compare_results.json")

if __name__ == "__main__":
    main()
