"""
AirDrawVocab Unified - Face Recognition/Auth Module

Chức năng:
- Phát hiện khuôn mặt bằng Haar Cascade của OpenCV.
- Đăng ký nhiều mẫu khuôn mặt cho mỗi người dùng.
- Huấn luyện và nhận diện bằng LBPHFaceRecognizer của opencv-contrib-python.
- Có cơ chế kiểm tra chất lượng, lấy nhiều mẫu, trung bình điểm để giảm nhận sai.

Lưu ý: LBPH là lựa chọn offline, nhẹ, phù hợp bài tập/demo local. Độ chính xác phụ thuộc ánh sáng,
góc mặt, số lượng mẫu đăng ký và camera. Nên đăng ký 30-50 mẫu/người trong điều kiện ánh sáng tốt.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
import time
import re

import cv2
import numpy as np


@dataclass
class FaceAuthResult:
    ok: bool
    message: str
    username: Optional[str] = None
    score: Optional[float] = None
    threshold: float = 70.0
    sample_count: int = 0


class FaceAuthManager:
    def __init__(self, base_dir: str | Path = "face_data", face_size: Tuple[int, int] = (200, 200), threshold: float = 70.0):
        self.base_dir = Path(base_dir)
        self.samples_dir = self.base_dir / "samples"
        self.model_path = self.base_dir / "lbph_face_model.yml"
        self.labels_path = self.base_dir / "labels.json"
        self.face_size = face_size
        self.threshold = float(threshold)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError("Không load được Haar Cascade nhận diện khuôn mặt của OpenCV.")

        self.recognizer = self._create_recognizer()

    @staticmethod
    def _create_recognizer():
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            raise RuntimeError(
                "Thiếu cv2.face.LBPHFaceRecognizer_create. Hãy cài opencv-contrib-python, "
                "không dùng opencv-python thường. Ví dụ: pip uninstall -y opencv-python && pip install opencv-contrib-python"
            )
        return cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)

    @staticmethod
    def _safe_name(username: str) -> str:
        username = username.strip()
        if not username:
            raise ValueError("Tên người dùng không được để trống.")
        return re.sub(r"[^a-zA-Z0-9_\-]+", "_", username)[:60]

    def _detect_largest_face(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=6,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return None
        return max(faces, key=lambda r: r[2] * r[3])

    def _preprocess_face(self, frame_bgr: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = face_box
        pad = int(0.12 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_bgr.shape[1], x + w + pad)
        y2 = min(frame_bgr.shape[0], y + h + pad)
        roi = frame_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.face_size, interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)
        return gray

    def _load_labels(self) -> Dict[int, str]:
        if not self.labels_path.exists():
            return {}
        data = json.loads(self.labels_path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in data.items()}

    def _save_labels(self, labels: Dict[int, str]) -> None:
        self.labels_path.write_text(json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_training_samples(self) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
        images: List[np.ndarray] = []
        ids: List[int] = []
        labels: Dict[int, str] = {}
        users = sorted([p for p in self.samples_dir.iterdir() if p.is_dir()])
        for label_id, user_dir in enumerate(users):
            labels[label_id] = user_dir.name
            for img_path in sorted(user_dir.glob("*.png")):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, self.face_size, interpolation=cv2.INTER_AREA)
                images.append(img)
                ids.append(label_id)
        return images, ids, labels

    def train(self) -> FaceAuthResult:
        images, ids, labels = self._read_training_samples()
        if len(images) < 5:
            return FaceAuthResult(False, "Cần ít nhất 5 mẫu khuôn mặt để huấn luyện.", sample_count=len(images), threshold=self.threshold)
        self.recognizer.train(images, np.array(ids, dtype=np.int32))
        self.recognizer.write(str(self.model_path))
        self._save_labels(labels)
        return FaceAuthResult(True, f"Đã huấn luyện nhận diện khuôn mặt với {len(images)} mẫu / {len(labels)} người.", sample_count=len(images), threshold=self.threshold)

    def _load_model(self) -> Tuple[bool, Dict[int, str]]:
        labels = self._load_labels()
        if not self.model_path.exists() or not labels:
            return False, labels
        self.recognizer.read(str(self.model_path))
        return True, labels

    def enroll_frame(self, username: str, frame_bgr: np.ndarray) -> FaceAuthResult:
        username = self._safe_name(username)
        face_box = self._detect_largest_face(frame_bgr)
        if face_box is None:
            return FaceAuthResult(False, "Không tìm thấy khuôn mặt rõ trong ảnh. Hãy nhìn thẳng camera và đủ sáng.", username=username, threshold=self.threshold)
        face = self._preprocess_face(frame_bgr, face_box)
        user_dir = self.samples_dir / username
        user_dir.mkdir(parents=True, exist_ok=True)
        idx = len(list(user_dir.glob("*.png"))) + 1
        cv2.imwrite(str(user_dir / f"sample_{idx:04d}.png"), face)
        train_result = self.train()
        sample_count = len(list(user_dir.glob("*.png")))
        return FaceAuthResult(True, f"Đã lưu mẫu khuôn mặt cho {username}. Tổng mẫu của người này: {sample_count}. {train_result.message}", username=username, threshold=self.threshold, sample_count=sample_count)

    def verify_frame(self, frame_bgr: np.ndarray, username: Optional[str] = None) -> FaceAuthResult:
        ready, labels = self._load_model()
        if not ready:
            return FaceAuthResult(False, "Chưa có model khuôn mặt. Hãy đăng ký khuôn mặt trước.", threshold=self.threshold)
        face_box = self._detect_largest_face(frame_bgr)
        if face_box is None:
            return FaceAuthResult(False, "Không tìm thấy khuôn mặt rõ trong ảnh.", threshold=self.threshold)
        face = self._preprocess_face(frame_bgr, face_box)
        pred_id, score = self.recognizer.predict(face)
        predicted_user = labels.get(int(pred_id), "unknown")
        ok = float(score) <= self.threshold
        if username:
            username = self._safe_name(username)
            ok = ok and predicted_user == username
        msg = (
            f"Xác thực {'THÀNH CÔNG' if ok else 'THẤT BẠI'}: dự đoán={predicted_user}, "
            f"score={float(score):.2f}, ngưỡng={self.threshold:.2f}."
        )
        return FaceAuthResult(ok, msg, username=predicted_user, score=float(score), threshold=self.threshold)

    def enroll_from_camera(self, username: str, camera_index: int = 0, samples: int = 35, delay: float = 0.08) -> FaceAuthResult:
        username = self._safe_name(username)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return FaceAuthResult(False, "Không mở được webcam để đăng ký khuôn mặt.", username=username, threshold=self.threshold)
        saved = 0
        user_dir = self.samples_dir / username
        user_dir.mkdir(parents=True, exist_ok=True)
        try:
            print("[FACE] Nhìn thẳng camera, xoay nhẹ trái/phải. Nhấn Q để dừng sớm.")
            while saved < samples:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                box = self._detect_largest_face(frame)
                preview = frame.copy()
                if box is not None:
                    x, y, w, h = box
                    cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    face = self._preprocess_face(frame, box)
                    idx = len(list(user_dir.glob("*.png"))) + 1
                    cv2.imwrite(str(user_dir / f"sample_{idx:04d}.png"), face)
                    saved += 1
                    time.sleep(delay)
                cv2.putText(preview, f"Enroll {username}: {saved}/{samples}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow("AirDrawVocab Face Enrollment", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        if saved < 5:
            return FaceAuthResult(False, f"Chỉ lưu được {saved} mẫu. Cần ít nhất 5 mẫu để train.", username=username, sample_count=saved, threshold=self.threshold)
        train_result = self.train()
        return FaceAuthResult(train_result.ok, f"Đăng ký {username}: lưu {saved} mẫu. {train_result.message}", username=username, sample_count=saved, threshold=self.threshold)

    def verify_from_camera(self, username: Optional[str] = None, camera_index: int = 0, samples: int = 12) -> FaceAuthResult:
        ready, labels = self._load_model()
        if not ready:
            return FaceAuthResult(False, "Chưa có model khuôn mặt. Hãy chạy: python face_cli.py enroll <ten_cua_ban>", threshold=self.threshold)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return FaceAuthResult(False, "Không mở được webcam để xác thực khuôn mặt.", threshold=self.threshold)
        votes: Dict[str, List[float]] = {}
        try:
            print("[FACE] Đang xác thực... nhìn thẳng camera. Nhấn Q để dừng.")
            while sum(len(v) for v in votes.values()) < samples:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                result = self.verify_frame(frame, username=None)
                preview = frame.copy()
                if result.username and result.score is not None:
                    votes.setdefault(result.username, []).append(result.score)
                    cv2.putText(preview, f"{result.username}: {result.score:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                else:
                    cv2.putText(preview, "No clear face", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("AirDrawVocab Face Login", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        if not votes:
            return FaceAuthResult(False, "Không thu được mẫu khuôn mặt hợp lệ để xác thực.", threshold=self.threshold)
        best_user, scores = min(votes.items(), key=lambda item: np.mean(item[1]))
        avg_score = float(np.mean(scores))
        ok = avg_score <= self.threshold
        if username:
            username = self._safe_name(username)
            ok = ok and best_user == username
        msg = f"Xác thực {'THÀNH CÔNG' if ok else 'THẤT BẠI'}: dự đoán={best_user}, score_trung_bình={avg_score:.2f}, ngưỡng={self.threshold:.2f}."
        return FaceAuthResult(ok, msg, username=best_user, score=avg_score, threshold=self.threshold, sample_count=sum(len(v) for v in votes.values()))

    def image_bytes_to_frame(self, image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("File gửi lên không phải ảnh hợp lệ.")
        return frame

    def enroll_image_bytes(self, username: str, image_bytes: bytes) -> FaceAuthResult:
        frame = self.image_bytes_to_frame(image_bytes)
        return self.enroll_frame(username, frame)

    def verify_image_bytes(self, image_bytes: bytes, username: Optional[str] = None) -> FaceAuthResult:
        frame = self.image_bytes_to_frame(image_bytes)
        return self.verify_frame(frame, username=username)
