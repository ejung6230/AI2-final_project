import cv2

print("카메라 테스트 시작...")

for i in range(5):
    print(f"\n카메라 인덱스 {i} 테스트 중...")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"✅ 카메라 인덱스 {i} 사용 가능")
        ret, frame = cap.read()
        print("프레임 읽기:", ret)
        cap.release()
    else:
        print(f"❌ 카메라 인덱스 {i} 사용 불가")

print("\n테스트 완료!")
