# plot_results.py
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    with open("outputs/compare_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    A = data["A_no_code"]
    B = data["B_with_interpreter"]

    A_reward = np.array([x["cum_reward"] for x in A], dtype=float)
    B_reward = np.array([x["cum_reward"] for x in B], dtype=float)
    A_regret = np.array([x["cum_regret"] for x in A], dtype=float)
    B_regret = np.array([x["cum_regret"] for x in B], dtype=float)

    t = np.arange(A_reward.shape[1])

    def ms(x):
        return x.mean(axis=0), x.std(axis=0)

    a_mr, a_sr = ms(A_reward)
    b_mr, b_sr = ms(B_reward)
    a_mg, a_sg = ms(A_regret)
    b_mg, b_sg = ms(B_regret)

    plt.figure(figsize=(12, 9))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, a_mr, 'r--', label='A: No Code')
    ax1.fill_between(t, a_mr-a_sr, a_mr+a_sr, color='r', alpha=0.15)
    ax1.plot(t, b_mr, 'g-', label='B: Interpreter+Bandit Opt')
    ax1.fill_between(t, b_mr-b_sr, b_mr+b_sr, color='g', alpha=0.15)
    ax1.set_title("Cumulative Reward (Higher Better)")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Cum Reward")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t, a_mg, 'r--', label='A: No Code')
    ax2.fill_between(t, a_mg-a_sg, a_mg+a_sg, color='r', alpha=0.15)
    ax2.plot(t, b_mg, 'g-', label='B: Interpreter+Bandit Opt')
    ax2.fill_between(t, b_mg-b_sg, b_mg+b_sg, color='g', alpha=0.15)
    ax2.set_title("Cumulative Regret (Lower Better)")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Cum Regret")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(t, a_mr/(t+1), 'r--', label='A avg')
    ax3.plot(t, b_mr/(t+1), 'g-', label='B avg')
    ax3.set_title("Average Reward Per Step")
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("Avg Reward")
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    a_final = A_reward[:, -1]
    b_final = B_reward[:, -1]
    ax4.bar(["A No Code", "B Interpreter"], [a_final.mean(), b_final.mean()],
            yerr=[a_final.std(), b_final.std()], color=["#ff9999", "#99cc99"])
    ax4.set_title("Final Cum Reward (mean±std)")
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/compare_plot.png", dpi=150)
    print("Saved: outputs/compare_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
