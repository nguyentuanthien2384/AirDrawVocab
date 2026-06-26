# AirDrawVocab Self-improving Loop storage

Thu muc nay duoc backend tao va cap nhat tu dong cho khu vuc **Self-improving Loop**.

## Cau truc

- `samples/stroke_samples.jsonl`: append-only log moi mau ve duoc luu qua `/game/stroke`.
- `samples/by_label/<label>/<sample_id>.json`: ban chi tiet tung mau theo nhan.
- `exports/latest/`: ban export moi nhat dung cho nut **Export data** va link download.
- `exports/<timestamp>_export/`: snapshot rieng cho moi lan export, khong bi ghi de.
- `jobs/<timestamp>_<mode>/`: log, metadata va snapshot model cua tung lan **Train stroke** hoac **Train image**.
- `status/retrain_status.json`: trang thai retrain moi nhat.
- `actions/events.jsonl`: nhat ky thao tac: luu mau, export, train, reload model.

Co the xoa cac snapshot cu neu can tiet kiem dung luong; khong nen xoa database `data/airdrawvocab_app.sqlite3`.
