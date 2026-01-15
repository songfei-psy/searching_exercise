# 计算和可视化实验指标

import json
import matplotlib.pyplot as plt


def compute_metrics(logs):
    n = len(logs)
    success_rate = sum(ep["success"] for ep in logs) / n
    avg_reward = sum(ep["reward"] for ep in logs) / n
    avg_steps = sum(ep["steps"] for ep in logs) / n
    avg_time = sum(ep["time"] for ep in logs) / n

    return {
        "episodes": n,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_time": avg_time,
    }


def _annotate_bars(ax):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def plot_results(file_paths, labels, figsize=(12, 8)):
    """
    基于保存的聚合指标，绘制更丰富的对比图：
    - 子图1：成功率
    - 子图2：平均回报
    - 子图3：平均步数（越低越好）
    - 子图4：平均用时（秒）

    备注：由于保存的是聚合指标，暂无法绘制置信区间。
    """
    metrics_list = []
    for path in file_paths:
        with open(path, "r") as f:
            metrics_list.append(json.load(f))

    success = [m.get("success_rate", 0.0) for m in metrics_list]
    reward = [m.get("avg_reward", 0.0) for m in metrics_list]
    steps = [m.get("avg_steps", 0.0) for m in metrics_list]
    times = [m.get("avg_time", 0.0) for m in metrics_list]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # 成功率
    ax1.bar(labels, success, color="#4CAF50")
    ax1.set_title("Success Rate")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Rate")
    _annotate_bars(ax1)

    # 平均回报
    ax2.bar(labels, reward, color="#2196F3")
    ax2.set_title("Average Reward")
    _annotate_bars(ax2)

    # 平均步数（越低越好）
    ax3.bar(labels, steps, color="#FF9800")
    ax3.set_title("Average Steps (lower is better)")
    _annotate_bars(ax3)

    # 平均用时
    ax4.bar(labels, times, color="#9C27B0")
    ax4.set_title("Average Time (s)")
    _annotate_bars(ax4)

    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle("Agent Performance Comparison", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # 文本小结
    summary_lines = ["Summary:"]
    for label, s, r, st, t in zip(labels, success, reward, steps, times):
        summary_lines.append(
            f"- {label}: success={s:.2f}, reward={r:.2f}, steps={st:.1f}, time={t:.3f}s"
        )
    print("\n".join(summary_lines))

    plt.show()
