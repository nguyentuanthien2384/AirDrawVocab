import os
import subprocess
import sys


def run(cmd):
    print("\n$ " + " ".join(cmd))
    return subprocess.call(cmd)


def main():
    print("=" * 60)
    print("AirDrawVocab Unified AI")
    print("1. Chạy game desktop")
    print("2. Chạy game desktop + bắt buộc xác thực khuôn mặt")
    print("3. Đăng ký khuôn mặt")
    print("4. Xác thực khuôn mặt thử")
    print("5. Chạy web chatbot API")
    print("=" * 60)
    choice = input("Chọn chức năng: ").strip()

    if choice == "1":
        return run([sys.executable, "game.py"])
    if choice == "2":
        username = input("Tên người dùng cần xác thực (bỏ trống nếu tự nhận diện): ").strip()
        cmd = [sys.executable, "game.py", "--face-login"]
        if username:
            cmd += ["--face-user", username]
        return run(cmd)
    if choice == "3":
        username = input("Nhập tên người dùng: ").strip()
        return run([sys.executable, "face_cli.py", "enroll", username])
    if choice == "4":
        username = input("Tên cần kiểm tra (bỏ trống nếu tự nhận diện): ").strip()
        cmd = [sys.executable, "face_cli.py", "verify"]
        if username:
            cmd += ["--username", username]
        return run(cmd)
    if choice == "5":
        print("Mở trình duyệt: http://127.0.0.1:8000")
        return run([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])

    print("Lựa chọn không hợp lệ.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
