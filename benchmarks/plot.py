"""Generate results/learning_curve.png from the latest benchmark JSON."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_RESULTS_DIR = Path(__file__).parent.parent / "results"


def _latest_benchmark() -> Path:
    files = sorted(_RESULTS_DIR.glob("benchmark_*.json"))
    if not files:
        raise FileNotFoundError(f"No benchmark JSON found in {_RESULTS_DIR}")
    return files[-1]


def plot(path: Path | None = None) -> Path:
    src = path or _latest_benchmark()
    data = json.loads(src.read_text(encoding="utf-8"))

    epochs = [e["epoch"] for e in data["per_epoch"]]
    memory_train = [e["memory_train_accuracy"] * 100 for e in data["per_epoch"]]
    holdout = [e["memory_holdout_accuracy"] * 100 for e in data["per_epoch"]]
    vanilla_acc = data["vanilla_train_accuracy"] * 100

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, memory_train, color="#1f77b4", linewidth=2.5, marker="o",
            markersize=5, label="Memory Agent (train)")
    ax.plot(epochs, holdout, color="#aec7e8", linewidth=2, marker="s",
            markersize=4, linestyle="--", label="Memory Agent (holdout)")
    ax.axhline(vanilla_acc, color="#7f7f7f", linewidth=2, linestyle=":",
               label=f"Vanilla baseline ({vanilla_acc:.1f}%)")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Three-Memory Agent vs Vanilla Baseline — Northwind Text-to-SQL", fontsize=13)
    ax.set_xticks(epochs)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    out = _RESULTS_DIR / "learning_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    plot(src)
