# Tải dataset TrashNet (ảnh rác, nền trắng) từ GitHub và giải nén vào data/
import io, os, shutil, urllib.request, zipfile

URL = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]  # bỏ lớp "trash" (quá ít ảnh, không rõ ràng)

os.makedirs("data", exist_ok=True)
print("Downloading TrashNet (~43 MB)...")
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
for name in z.namelist():
    parts = name.split("/")
    # chỉ lấy file ảnh .jpg thuộc 5 lớp cần dùng
    if len(parts) == 3 and parts[1] in CLASSES and parts[2].endswith(".jpg"):
        dst = os.path.join("data", parts[1], parts[2])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with z.open(name) as src, open(dst, "wb") as f:
            shutil.copyfileobj(src, f)
for c in CLASSES:
    print(f"{c:10s} {len(os.listdir(os.path.join('data', c)))} images")
