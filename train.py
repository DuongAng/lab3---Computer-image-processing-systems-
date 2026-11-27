# Train MobileNetV3 phân loại rác 5 lớp trên TrashNet, đánh giá trên tập val
import argparse, json, os, time
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from common import CLASSES, build_model, device, eval_tf, train_tf
from plots import plot_cm


def run_epoch(model, loader, loss_fn, dev, opt=None):
    # opt != None -> train, ngược lại -> chỉ đánh giá
    model.train(opt is not None)
    tot_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = loss_fn(out, y)
            if opt:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(y)
            preds += out.argmax(1).tolist(); labels += y.tolist()
    preds, labels = np.array(preds), np.array(labels)
    return tot_loss / len(labels), (preds == labels).mean(), preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weights", default=None, help="file weights ImageNet (nếu không tải tự động được)")
    a = ap.parse_args()
    torch.manual_seed(0)
    os.makedirs("results", exist_ok=True)
    dev = device()

    # Chia 80% train / 20% val, giữ tỉ lệ các lớp (stratify), seed cố định
    full = ImageFolder("data")
    assert full.classes == CLASSES, full.classes
    idx = np.arange(len(full))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=full.targets, random_state=0)
    train_ds = Subset(ImageFolder("data", train_tf), tr_idx)
    val_ds = Subset(ImageFolder("data", eval_tf), va_idx)
    tl = DataLoader(train_ds, a.bs, shuffle=True, num_workers=2)
    vl = DataLoader(val_ds, a.bs, num_workers=2)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Device: {dev}")

    model = build_model(True, a.weights).to(dev)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    hist, best = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        trl, tra, _, _ = run_epoch(model, tl, loss_fn, dev, opt)
        val, vaa, _, _ = run_epoch(model, vl, loss_fn, dev)
        sched.step()
        for k, v in zip(hist, (trl, val, tra, vaa)):
            hist[k].append(float(v))
        print(f"Epoch {ep:2d}/{a.epochs}  train loss {trl:.3f} acc {tra:.3f} | "
              f"val loss {val:.3f} acc {vaa:.3f}  ({time.time() - t0:.0f}s)")
        if vaa > best:  # lưu model tốt nhất theo val accuracy
            best = vaa
            torch.save(model.state_dict(), "model.pth")
    print(f"Total training time: {time.time() - t0:.0f}s")

    # Đánh giá lại model tốt nhất trên val
    model.load_state_dict(torch.load("model.pth", map_location=dev))
    _, acc, p, y = run_epoch(model, vl, loss_fn, dev)
    f1 = f1_score(y, p, average="macro")
    print(f"\nBest model on VAL: accuracy {acc:.4f}, macro-F1 {f1:.4f}")
    print(classification_report(y, p, target_names=CLASSES, digits=3))
    plot_cm(confusion_matrix(y, p), f"Validation (acc={acc:.3f})", "results/cm_val.png")
    json.dump(hist, open("results/history.json", "w"), indent=1)

    # Đồ thị loss / accuracy theo epoch -> xem có overfit không
    e = range(1, a.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(e, hist["train_loss"], "o-", label="train"); ax[0].plot(e, hist["val_loss"], "o-", label="val")
    ax[1].plot(e, hist["train_acc"], "o-", label="train"); ax[1].plot(e, hist["val_acc"], "o-", label="val")
    for x_, t in zip(ax, ("Loss (CrossEntropy)", "Accuracy")):
        x_.set_title(t); x_.set_xlabel("epoch"); x_.grid(alpha=.3); x_.legend()
    plt.tight_layout(); plt.savefig("results/curves.png", dpi=120)


if __name__ == "__main__":
    main()
