"""Reproduce the scientific Top1 score distribution figure from raw results."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
    statistics = []
    for ax, mode in zip(axes, ["vector", "hybrid", "rerank"]):
        rows = json.loads(
            (ROOT / f"experiments/results/retrieval_{mode}.json").read_text(
                encoding="utf-8"
            )
        )["rows"]
        for answerable, label in [(True, "可回答 n=60"), (False, "库外 n=8")]:
            scores = [
                r["top5"][0]["score"]
                for r in rows
                if r["answerable"] == answerable and r["top5"]
            ]
            ax.hist(scores, bins=np.linspace(0.4, 0.8, 17), alpha=0.6, label=label)
            statistics.append(
                dict(
                    mode=mode,
                    answerable=answerable,
                    n=len(scores),
                    mean=float(np.mean(scores)),
                    min=min(scores),
                    max=max(scores),
                )
            )
        ax.set_title(mode)
        ax.set_xlabel("Top1 分数")
        ax.set_ylabel("题目数")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Top1 分数分布  相似度与重排分数不可跨模式校准", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/assets/top1_distribution.png", dpi=150)
    plt.close(fig)
    (ROOT / "experiments/results/top1_distribution.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
