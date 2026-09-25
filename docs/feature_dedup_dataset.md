# MobileNet feature dedup dataset

Script `tools/build_feature_dedup_datasets.py` dung MobileNetV3-Small pretrained
de trich embedding cua tung crop nguoi va cosine similarity de loai crop gan trung.

```text
frame -> detect/classify tat ca nguoi
      -> dataset_yolo: luu anh + tat ca bbox
      -> dataset_cls: MobileNet embedding tung crop
                    -> ghep hoc sinh bang IoU bbox
                    -> chi luu crop moi/khac
```

Lenh mau:

```powershell
python tools\build_feature_dedup_datasets.py video.mp4 --device 0 --frame-stride 3 --similarity-threshold 0.985 --visualize
```

Nhieu video trong mot lenh:

```powershell
python tools\build_feature_dedup_datasets.py video01.mp4 video02.mp4 video03.mp4 --device 0 --frame-stride 3 --visualize
```

Hoac truyen ca thu muc; script se tim video trong cac thu muc con:

```powershell
python tools\build_feature_dedup_datasets.py D:\videos_ca_sang D:\videos_ca_chieu --device 0 --frame-stride 3 --visualize
```

Moi model chi duoc nap mot lan. Cac video duoc xu ly lan luot tren cung GPU va ket
qua duoc gop vao mot thu muc output, voi ten file bat dau bang ten video nguon.

MobileNet va cac model YOLO dung chung thiet bi tu `--device`. Lan chay dau tien,
torchvision co the tai weights pretrained neu weights chua nam trong cache.

Nguong cao hon se giu nhieu frame hon:

- `0.995`: loc nhe
- `0.985`: mac dinh
- `0.970`: loc manh

`dataset_yolo` khong bi loc boi MobileNet: moi frame co nguoi va tat ca bbox deu
duoc luu. Chi `dataset_cls` bo crop gan trung cua cung hoc sinh. Crop co nhan thay
doi hoac confidence thap luon duoc giu. `metadata.csv` co `crop_saved=1/0`, ly do
va cosine similarity de kiem tra.

`frames.csv` co mot dong cho moi anh YOLO, gom ten frame, file label, so object,
danh sach class va thong ke `normal/sleep/raisehand`. `metadata.csv` van luu chi
tiet tung bbox va tung crop classification.

Khi bat dau, script in duong dan va `names` cua ca ba model. Sau moi video, log
`VIDEO DONE` thong ke frame, bbox, raw class ID cua model sleep/raisehand, nhan
cuoi cung va so crop classification da luu/bo. Neu `--raisehand-positive-id`
khong ton tai trong model, script in canh bao ngay.

Model raise-hand dung raw class `0=normal`, `1=raisehand`. Khi tao dataset, raw
class `1` duoc anh xa thanh class YOLO `2=raisehand`.
