# MobileNet feature dedup dataset

Script `tools/build_feature_dedup_datasets.py` dung MobileNetV3-Small pretrained
de trich embedding cua toan frame va cosine similarity de loai frame gan trung.

```text
frame -> detect/classify tat ca nguoi -> MobileNet embedding
      -> so voi cac frame da luu co cung chu ky bbox/nhan
      -> neu khac: luu anh + tat ca bbox + tat ca crop
```

Lenh mau:

```powershell
python tools\build_feature_dedup_datasets.py video.mp4 --device 0 --frame-stride 3 --similarity-threshold 0.985 --visualize
```

MobileNet va cac model YOLO dung chung thiet bi tu `--device`. Lan chay dau tien,
torchvision co the tai weights pretrained neu weights chua nam trong cache.

Nguong cao hon se giu nhieu frame hon:

- `0.995`: loc nhe
- `0.985`: mac dinh
- `0.970`: loc manh

Frame co bo cuc bbox/nhan thay doi hoac co confidence thap luon duoc giu. Khi mot
frame duoc giu, file YOLO chua day du moi bbox `visible-person` cua frame do.
