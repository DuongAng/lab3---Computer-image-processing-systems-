# Phần dùng chung cho train.py và test.py
import timm, torch
from torchvision import transforms as T

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # chuẩn hoá theo ImageNet

# Ảnh train: biến đổi ngẫu nhiên (cắt, lật, xoay, đổi màu) để mạng không học thuộc lòng
train_tf = T.Compose([
    T.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
    T.ColorJitter(0.3, 0.3, 0.3, 0.05),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
# Ảnh val/test: chỉ resize + cắt giữa
eval_tf = T.Compose([
    T.Resize(SIZE), T.CenterCrop(SIZE),
    T.ToTensor(), T.Normalize(MEAN, STD),
])


def build_model(pretrained=True, weights_file=None):
    # MobileNetV3-Large đã học trên ImageNet, thay lớp cuối bằng 5 đầu ra
    overlay = dict(file=weights_file) if weights_file else None
    return timm.create_model("mobilenetv3_large_100", pretrained=pretrained,
                             num_classes=len(CLASSES), pretrained_cfg_overlay=overlay)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
