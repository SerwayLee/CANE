import cv2
import numpy as np
import time
import torch
from pathlib import Path
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from ultralytics import YOLO

# ==============================
# 1. 사용자 설정(Variables)
# ==============================
CAM_URL               = "http://192.168.0.101:8080/video"  # 단일 IP Webcam URL
BASE_DIR              = Path(__file__).parent.resolve()
FINAL_MODEL           = BASE_DIR / 'runs' / 'coco_pv_yolov8n' / 'weights' / 'best_final.pt'

FRAME_INTERVAL        = 0.1  # 프레임 처리 간격 (초)
CAUTION_TTC_THRESHOLD = 4.0  # TTC 주의 임계 (초)
DANGER_TTC_THRESHOLD  = 2.0  # TTC 위험 임계 (초)
SAFE_FRAME_BUFFER     = 3    # 안전 히스테리시스 프레임 수

# 허용할 클래스 총 13개
ALLOWED_CLASSES = {
    'person','bicycle','car','motorcycle',
    'airplane','bus','train','truck','boat',
    'beam','deer','gcooter','others'
}

# ==============================
# 2. MiDaS Monocular Depth 모델 로드
# ==============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').to(device).eval()
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform

def estimate_depth(frame):
    """
    모노큘러(relative) depth map 추정
    TTC 계산 식:
      TTC = d_curr * Δt / (d_prev - d_curr)
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = midas_transforms(img).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=frame.shape[:2], mode='bilinear', align_corners=False
        ).squeeze()
    return pred.cpu().numpy().astype(np.float32)

# ==============================
# 3. YOLOv8 객체 검출 및 필터링
# ==============================
def detect_objects(frame, model):
    results = model(frame)[0]
    dets = []
    if not results.boxes:
        return dets
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        name   = model.names.get(cls_id, str(cls_id))
        if name not in ALLOWED_CLASSES:
            continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cx,cy = (x1+x2)/2.0, (y1+y2)/2.0
        dets.append({
            'class_name': name,
            'bbox': (x1,y1,x2,y2),
            'center': (cx,cy),
            'prev_depth': None,
            'curr_depth': None
        })
    return dets

# ==============================
# 4. 광학 흐름 기반 트래킹 (Depth 연결)
# ==============================
def track_object_movements(prev_gray, curr_gray, prev_dets, curr_dets):
    prev_pts = np.array([d['center'] for d in prev_dets], dtype=np.float32).reshape(-1,1,2)
    if len(prev_pts) > 0:
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    else:
        curr_pts, status = [], []
    used = [False]*len(curr_dets)
    for i, pd in enumerate(prev_dets):
        if i>=len(status) or status[i]==0:
            continue
        nx,ny = curr_pts[i][0]
        cls = pd['class_name']
        best_j, min_dist = None, float('inf')
        for j, cd in enumerate(curr_dets):
            if used[j] or cd['class_name']!=cls:
                continue
            cx,cy = cd['center']
            dist = np.hypot(cx-nx, cy-ny)
            if dist<min_dist and dist<50:
                best_j, min_dist = j, dist
        if best_j is not None:
            curr_dets[best_j]['prev_depth'] = pd['curr_depth']
            curr_dets[best_j]['curr_depth'] = curr_dets[best_j]['curr_depth']
            used[best_j] = True
    for cd in curr_dets:
        if cd['prev_depth'] is None:
            cd['prev_depth'] = cd['curr_depth']
    return curr_dets

# ==============================
# 5. 충돌 위험 평가 (TTC 계산)
# ==============================
def evaluate_collision_risk(dets):
    immediate, caution, obj = False, False, None
    for d in dets:
        dp, dc = d['prev_depth'], d['curr_depth']
        if dp is None or dc is None or dp<=dc:
            continue
        # 상대 깊이만으로도 TTC 계산 가능
        ttc = (dc * FRAME_INTERVAL) / (dp - dc)
        if ttc < DANGER_TTC_THRESHOLD:
            immediate, obj = True, d['class_name']
            break
        elif ttc < CAUTION_TTC_THRESHOLD:
            caution = True
            if obj is None:
                obj = d['class_name']
    return immediate, caution, obj

# ==============================
# 6. 경고 메시지 생성 및 출력 (print)
# ==============================
def generate_warning_message(obj, level):
    # beam, deer, gcooter, others는 scooter로 통일
    out = obj if obj=='person' else 'vehicle'
    if level=='danger':
        return f"Warning: A {out} is approaching fast! Take immediate action!"
    return f"Caution: A {out} is approaching. Please be aware."

# ==============================
# 7. 메인 루프
# ==============================
def main():
    cap = cv2.VideoCapture(CAM_URL)
    if not cap.isOpened():
        print("Error: Camera open failed.")
        return
    # YOLO 모델 로드
    yolo = YOLO(str(FINAL_MODEL))

    prev_gray = None
    prev_dets = []
    caution_count = 0
    danger_announced = False
    caution_announced = False
    safe_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 1) 상대 depth 맵 추정
            depth_map = estimate_depth(frame)
            # 2) 객체 검출
            curr_dets = detect_objects(frame, yolo)
            # 3) curr_depth 채우기
            for cd in curr_dets:
                x,y = map(int, cd['center'])
                cd['curr_depth'] = float(depth_map[y, x])
            # 4) 트래킹 & depth 연결
            if prev_gray is not None:
                curr_dets = track_object_movements(prev_gray, gray, prev_dets, curr_dets)
            # 5) 충돌 위험 판정
            imm, caution, obj = evaluate_collision_risk(curr_dets)
            # 6) 히스테리시스 + 메시지 출력
            if imm:
                msg = generate_warning_message(obj, 'danger')
                if not danger_announced:
                    print(msg)
                    danger_announced = True
                caution_count = 0
                caution_announced = False
                safe_count = 0
            elif caution:
                msg = generate_warning_message(obj, 'caution')
                caution_count += 1
                safe_count = 0
                if caution_count >= 5 and not caution_announced:
                    print(msg)
                    caution_announced = True
                danger_announced = False
            else:
                safe_count += 1
                danger_announced = False
                if safe_count >= SAFE_FRAME_BUFFER:
                    caution_count = 0
                    caution_announced = False
                    safe_count = 0
            # 7) 상태 갱신
            prev_gray = gray.copy()
            prev_dets = curr_dets
            time.sleep(FRAME_INTERVAL)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
