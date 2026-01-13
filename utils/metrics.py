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


def plot_results(file_paths, labels):
    for path, label in zip(file_paths, labels):
        with open(path, "r") as f:
            data = json.load(f)
        plt.bar(label, data["success_rate"])
    plt.ylabel("Success Rate")
    plt.title("Agent Performance Comparison")
    plt.show()
