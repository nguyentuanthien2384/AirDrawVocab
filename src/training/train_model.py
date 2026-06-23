"""
AirDrawVocab - Huấn luyện mô hình CNN cơ bản.
Tải dữ liệu QuickDraw và huấn luyện model CNN 2 lớp.
"""
import csv, json, os, urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.size': 12, 'axes.titlesize': 14,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3,
})
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from config import (
    CATEGORIES, NUM_CLASSES, RANDOM_STATE,
    DATA_DIR, RESULTS_DIR, REPORTS_DIR
)
from data_utils import load_dataset, split_dataset

print(f"TensorFlow: {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU')}")

# === CẤU HÌNH ===
IMG_SIZE = 28
BATCH_SIZE = 32
EPOCHS = 30
MODEL_SAVE_PATH = 'models/airdrawvocab_retrained_model.h5'
BEST_MODEL_PATH = 'models/airdrawvocab_best_model.h5'

os.makedirs(str(RESULTS_DIR), exist_ok=True)
os.makedirs(str(REPORTS_DIR), exist_ok=True)

# === BƯỚC 1: DOWNLOAD DỮ LIỆU (nếu chưa có) ===
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for category in CATEGORIES:
        filepath = DATA_DIR / f'{category}.npy'
        if not filepath.exists():
            url = BASE_URL + urllib.request.quote(f'{category}.npy')
            print(f'  Đang tải: {category}...', end=' ')
            urllib.request.urlretrieve(url, filepath)
            print(f'✓ ({filepath.stat().st_size / 1024 / 1024:.1f} MB)')

# === BƯỚC 2: LOAD & CHIA DỮ LIỆU ===
print("\n" + "=" * 60)
print("LOAD VÀ CHIA DỮ LIỆU (Train 800 / Val 150 / Test 150)")
print("=" * 60)

X, y = load_dataset()
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, seed=RANDOM_STATE)

# Reshape: [N, 28, 28, 1]
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat = to_categorical(y_val, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)

print(f'  Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}')

# === MẪU DỮ LIỆU ===
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle('Mẫu hình vẽ từ QuickDraw Dataset', fontsize=16, fontweight='bold')
for idx, (ax, cat) in enumerate(zip(axes.flat, CATEGORIES)):
    sample = X_train[y_train == idx][0].reshape(IMG_SIZE, IMG_SIZE)
    ax.imshow(sample, cmap='gray_r'); ax.set_title(cat); ax.axis('off')
if len(CATEGORIES) < 20:
    axes.flat[-1].axis('off')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'sample_drawings.png', dpi=300, bbox_inches='tight')
plt.close()

# === BƯỚC 3: XÂY DỰNG MODEL CNN ===
print("\n" + "=" * 60)
print("""Kiến trúc CNN:
  Input (28x28x1)
  → Conv2D(16, 3x3, ReLU) + BatchNorm → MaxPool(2x2) + Dropout(0.15)
  → Conv2D(32, 3x3, ReLU) + BatchNorm → MaxPool(2x2) + Dropout(0.25)
  → Flatten → Dense(64, ReLU) + Dropout(0.40) → Dense(NUM_CLASSES, Softmax)
""")

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(), MaxPooling2D((2, 2)), Dropout(0.15),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(), MaxPooling2D((2, 2)), Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'), Dropout(0.40),
    Dense(NUM_CLASSES, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === BƯỚC 4: HUẤN LUYỆN ===
callbacks = [
    ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    CSVLogger(str(REPORTS_DIR / 'training_log.csv')),
]

history = model.fit(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE, epochs=EPOCHS,
    validation_data=(X_val, y_val_cat),
    callbacks=callbacks, verbose=1
)

# === BƯỚC 5: BIỂU ĐỒ TRAINING ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Train'); ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy', fontweight='bold'); ax1.legend()
ax2.plot(history.history['loss'], label='Train'); ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Loss', fontweight='bold'); ax2.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
plt.close()

# === BƯỚC 6: ĐÁNH GIÁ ===
print("\n" + "=" * 60 + "\nĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST\n" + "=" * 60)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
mae = mean_absolute_error(to_categorical(y_test, NUM_CLASSES), y_pred_probs)
mse = mean_squared_error(to_categorical(y_test, NUM_CLASSES), y_pred_probs)

print(f"""
╔══════════════════════════════════════════╗
║       BẢNG KẾT QUẢ ĐÁNH GIÁ            ║
╠══════════════════════════════════════════╣
║  Test Accuracy:       {test_acc*100:>7.2f}%          ║
║  Precision:           {precision:>10.4f}          ║
║  Recall:              {recall:>10.4f}          ║
║  F1-Score:            {f1:>10.4f}          ║
║  MAE:                 {mae:>10.4f}          ║
║  MSE:                 {mse:>10.4f}          ║
╚══════════════════════════════════════════╝
""")

report_text = classification_report(y_test, y_pred, target_names=CATEGORIES)
report_dict = classification_report(y_test, y_pred, target_names=CATEGORIES, output_dict=True)
print(report_text)

with open(REPORTS_DIR / 'classification_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)
with open(REPORTS_DIR / 'classification_report.json', 'w', encoding='utf-8') as f:
    json.dump(report_dict, f, ensure_ascii=False, indent=2)
with open(REPORTS_DIR / 'metrics_summary.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric', 'value'])
    for k, v in [('test_accuracy', accuracy), ('precision', precision), ('recall', recall),
                 ('f1', f1), ('mae', mae), ('mse', mse)]:
        w.writerow([k, v])

# === BƯỚC 7: CONFUSION MATRIX ===
cm = confusion_matrix(y_test, y_pred)
np.savetxt(REPORTS_DIR / 'confusion_matrix.csv', cm, delimiter=',', fmt='%d')
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Error analysis
wrong = np.where(y_test != y_pred)[0]
with open(REPORTS_DIR / 'error_analysis.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['test_index', 'true_label', 'predicted_label', 'confidence'])
    for i in wrong:
        w.writerow([int(i), CATEGORIES[int(y_test[i])], CATEGORIES[int(y_pred[i])], f"{np.max(y_pred_probs[i]):.4f}"])

# === BƯỚC 8: DỰ ĐOÁN MẪU ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
img = X_test[0].reshape(IMG_SIZE, IMG_SIZE)
probs = y_pred_probs[0]
ax1.imshow(img, cmap='gray_r'); ax1.set_title(f'True: {CATEGORIES[y_test[0]]}', fontweight='bold'); ax1.axis('off')
colors = ['tab:red' if i == np.argmax(probs) else 'tab:blue' for i in range(NUM_CLASSES)]
ax2.barh(CATEGORIES, probs, color=colors)
ax2.set_title(f'Predicted: {CATEGORIES[np.argmax(probs)]} ({np.max(probs):.2f})', fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'prediction_sample.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Kết quả dự đoán trên tập Test', fontsize=16, fontweight='bold')
for ax, i in zip(axes.flat, np.random.RandomState(42).choice(len(X_test), 10, replace=False)):
    ax.imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray_r')
    tl, pl = CATEGORIES[y_test[i]], CATEGORIES[y_pred[i]]
    ax.set_title(f'True: {tl}\nPred: {pl}', fontsize=10, color='green' if tl == pl else 'red', fontweight='bold')
    ax.axis('off')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'multiple_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

# === BƯỚC 9: LƯU MODEL ===
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
with open('models/categories.json', 'w', encoding='utf-8') as f:
    json.dump(CATEGORIES, f, ensure_ascii=False, indent=2)

print(f"""
{'='*60}
🎉 HOÀN THÀNH HUẤN LUYỆN!
  Model: {MODEL_SAVE_PATH} ({os.path.getsize(MODEL_SAVE_PATH)/1024:.1f} KB)
  Reports: {REPORTS_DIR}/
  Results: {RESULTS_DIR}/
{'='*60}
""")
