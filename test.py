# Kiểm tra model trên tập test tự chụp (thư mục test/<lớp>/*.jpg)
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision.datasets import ImageFolder
from common import CLASSES, build_model, device, eval_tf
from plots import plot_cm


def load(path):
    # Ảnh điện thoại hay bị xoay theo EXIF -> xoay về đúng chiều
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def main():
    dev = device()
    model = build_model(pretrained=False).to(dev).eval()
    model.load_state_dict(torch.load("model.pth", map_location=dev))
    ds = ImageFolder("test", loader=load)
    assert ds.classes == CLASSES, ds.classes
    print(f"Test images: {len(ds)}  Device: {dev}")

    preds, probs, labels, imgs, t = [], [], [], [], 0.0
    with torch.no_grad():
        for img, y in ds:
            x = eval_tf(img).unsqueeze(0).to(dev)
            t0 = time.perf_counter()
            pr = model(x).softmax(1)[0].cpu()
            t += time.perf_counter() - t0
            preds.append(int(pr.argmax())); probs.append(float(pr.max())); labels.append(y)
            imgs.append(img.copy())
            imgs[-1].thumbnail((256, 256))
    p, y = np.array(preds), np.array(labels)
    acc, f1 = (p == y).mean(), f1_score(y, p, average="macro")
    print(f"TEST: accuracy {acc:.4f}, macro-F1 {f1:.4f}")
    print(classification_report(y, p, target_names=CLASSES, digits=3, zero_division=0))
    print(f"Mean inference time: {1000 * t / len(ds):.1f} ms/image")
    plot_cm(confusion_matrix(y, p, labels=range(len(CLASSES))), f"Test (acc={acc:.3f})", "results/cm_test.png")

    # Lưới ảnh: tiêu đề xanh = đúng, đỏ = sai
    n = len(ds); cols = 8; rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.25))
    for i, ax in enumerate(np.ravel(axs)):
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i])
            ok = p[i] == y[i]
            ax.set_title(f"{CLASSES[p[i]]} {probs[i]:.2f}" + ("" if ok else f"\n(true: {CLASSES[y[i]]})"),
                         fontsize=8, color="green" if ok else "red")
    plt.tight_layout(); plt.savefig("results/test_grid.jpg", dpi=90)
    print("Saved results/cm_test.png, results/test_grid.jpg")


if __name__ == "__main__":
    main()
