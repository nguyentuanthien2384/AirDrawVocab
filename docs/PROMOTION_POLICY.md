# PROMOTION_POLICY.md

## Nguyên tắc promote model

Không tự động thay model đang chạy chỉ vì train xong. Mỗi candidate phải đi qua 4 bước:

1. Train candidate.
2. Evaluate trên benchmark cố định.
3. So sánh với release hiện tại.
4. Promote nếu đạt gate.

## Gate tối thiểu

- Không giảm macro F1 so với model hiện tại.
- Không tăng nhầm lẫn ở top confusion pairs như `book ↔ door`, `book ↔ pants`, `envelope ↔ square`.
- Confidence phải được calibration hoặc ít nhất ghi rõ chưa calibration.
- Model phải load được bằng runtime layer.

## Lệnh promote

```bash
python src/training/promote_candidate.py \
  --candidate-image models/image_cnn_candidate.keras \
  --candidate-categories models/categories_candidate.json \
  --report assets/reports/releases/candidate/summary.json \
  --allow-if-weak-data
```

`--allow-if-weak-data` chỉ dành cho giai đoạn prototype. Khi có đủ dữ liệu thật, bỏ flag này.
