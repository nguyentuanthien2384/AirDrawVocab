import argparse
from face_auth import FaceAuthManager


def main():
    parser = argparse.ArgumentParser(description="AirDrawVocab face enrollment / verification")
    sub = parser.add_subparsers(dest="command", required=True)

    enroll = sub.add_parser("enroll", help="Đăng ký khuôn mặt bằng webcam")
    enroll.add_argument("username", help="Tên người dùng, ví dụ: thien")
    enroll.add_argument("--camera", type=int, default=0)
    enroll.add_argument("--samples", type=int, default=35)

    verify = sub.add_parser("verify", help="Xác thực khuôn mặt bằng webcam")
    verify.add_argument("--username", default=None, help="Tên cần kiểm tra; bỏ trống để tự nhận diện")
    verify.add_argument("--camera", type=int, default=0)
    verify.add_argument("--samples", type=int, default=12)

    args = parser.parse_args()
    manager = FaceAuthManager()

    if args.command == "enroll":
        result = manager.enroll_from_camera(args.username, camera_index=args.camera, samples=args.samples)
    else:
        result = manager.verify_from_camera(args.username, camera_index=args.camera, samples=args.samples)

    print(result.message)
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
