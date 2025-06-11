import cv2
from ultralytics import YOLO

# 1) YOLO 모델 로드
model = YOLO('path/to/your_model.pt')

# 2) IP Webcam 스트림 열기 (MJPEG)
stream_url = 'http://192.168.0.10:8080/video'  # IP Webcam 기본 MJPEG 엔드포인트
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    raise RuntimeError(f"스트림 열기 실패: {stream_url}")

# 3) 프레임 처리 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추론
    results = model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])
        label  = f"{model.names[cls_id]} {conf:.2f}"

        # 바운더리 박스 및 레이블
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # 결과 디스플레이
    cv2.imshow('Mobile YOLO Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()