# Vẽ ma trận nhầm lẫn (confusion matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import CLASSES


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)
