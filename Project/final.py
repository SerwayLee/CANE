import cv2
import numpy as np
import time
import torch
import threading
import requests
from pathlib import Path
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

# ==============================
# 1. 설정
# ==============================
CAM_URL               = "http://172.20.50.20:8080/video"
BASE_DIR              = Path(__file__).parent.resolve()
FINAL_MODEL           = BASE_DIR / "best_final.pt"
SERVER_URL            = "https://port-0-delta-1cupyg2klvgk9x51.sel5.cloudtype.app/receive"

FRAME_INTERVAL        = 0.10   # 처리 주기 목표 (s)
CAUTION_TTC_THRESHOLD = 7.0
DANGER_TTC_THRESHOLD  = 5.0
SAFE_FRAME_BUFFER     = 3
CONF_THRESHOLD        = 0.6

PROC_W, PROC_H = 320, 240        # 전처리 해상도

ALLOWED_CLASSES = {
    "person","bicycle","car","motorcycle","airplane","bus",
    "train","truck","boat","beam","deer","gcooter","others"
}

# ==============================
# 2. 모델 로드
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
midas_tf = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
yolo = YOLO(str(FINAL_MODEL))

# ==============================
# 3. 헬퍼 함수
# ==============================
def estimate_depth(frame_bgr):
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp  = midas_tf(rgb).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(PROC_H, PROC_W),
            mode="bilinear", align_corners=False
        ).squeeze()
    return pred.cpu().numpy().astype(np.float32)

def detect_objects(frame_bgr):
    results = yolo(frame_bgr)[0]
    dets = []
    if not results.boxes:
        return dets
    for box in results.boxes:
        conf = float(box.conf[0].item())
        if conf < CONF_THRESHOLD:  continue
        cls_id = int(box.cls[0].item())
        name   = yolo.names.get(cls_id, str(cls_id))
        if name not in ALLOWED_CLASSES:  continue
        if name != "person":
            name = "vehicle"

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dets.append({
            "class_name":  name,
            "confidence":  conf,
            "bbox":        (x1, y1, x2, y2),
            "prev_center": None,
            "center":      (cx, cy),
            "prev_depth":  None,
            "curr_depth":  None,
        })
    return dets

def track_with_lk(prev_gray, curr_gray, prev_dets, curr_dets):
    if not prev_dets:
        return curr_dets

    prev_pts = np.array([d["center"] for d in prev_dets], np.float32).reshape(-1,1,2)
    curr_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    used = [False] * len(curr_dets)
    for i, (pd, valid) in enumerate(zip(prev_dets, st.flatten())):
        if not valid:  continue
        nx, ny = curr_pts[i][0]
        cls = pd["class_name"]
        best_j, best_dist = None, float("inf")
        for j, cd in enumerate(curr_dets):
            if used[j] or cd["class_name"] != cls:  continue
            cx, cy = cd["center"]
            dist = np.hypot(cx - nx, cy - ny)
            if dist < best_dist and dist < 50:
                best_j, best_dist = j, dist
        if best_j is not None:
            cd = curr_dets[best_j]
            cd["prev_center"] = pd["center"]
            cd["prev_depth"]  = pd["curr_depth"]
            used[best_j] = True

    for cd in curr_dets:
        if cd["prev_center"] is None:
            cd["prev_center"] = cd["center"]
        if cd["prev_depth"] is None:
            cd["prev_depth"]  = cd["curr_depth"]
    return curr_dets

def evaluate_collision(dets):
    for d in dets:
        dp, dc = d["prev_depth"], d["curr_depth"]
        if dp is None or dc is None or dp <= dc:  continue
        ttc = (dc * FRAME_INTERVAL) / (dp - dc)
        if 0 < ttc < DANGER_TTC_THRESHOLD:
            return True, False, d
        if 0 < ttc < CAUTION_TTC_THRESHOLD:
            return False, True, d
    return False, False, None

def make_warning(det, level):
    px, _ = det["prev_center"]; cx, _ = det["center"]
    dx = cx - px
    if cx < PROC_W * 0.33:
        rel_pos, safe_dir = "left", "right"
    elif cx > PROC_W * 0.67:
        rel_pos, safe_dir = "right", "left"
    else:
        rel_pos, safe_dir = "front", "side"

    if dx < -5:
        moving = "from your left";  safe_dir = "right"
    elif dx > 5:
        moving = "from your right"; safe_dir = "left"
    else:
        moving = "ahead"

    obj = det["class_name"]
    if level == "danger":
        return f"⚠️ Warning: A {obj} is rapidly approaching {moving}! Step to your {safe_dir} now!"
    else:
        return f"Caution: A {obj} is approaching {moving}. Please move to your {safe_dir}."

# ==============================
# 4. 멀티스레드 파이프라인
# ==============================
def capture_worker(cap, q):
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (PROC_W, PROC_H))
        while not q.empty():
            try: q.get_nowait()
            except Empty: break
        q.put(frame)
    cap.release()

def processing_worker(q, stop_evt):
    prev_gray, prev_dets = None, []
    caution_cnt = safe_cnt = 0
    last_sent_ts = 0.0                        # ★ 경고 rate-limit
    with ThreadPoolExecutor(max_workers=4) as pool:
        while not stop_evt.is_set():
            try:
                frame = q.get(timeout=0.5)
            except Empty:
                continue
            loop_start = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            depth = estimate_depth(frame)
            curr_dets = detect_objects(frame)
            for d in curr_dets:
                x, y = map(int, d["center"])
                d["curr_depth"] = float(depth[y, x])

            if prev_gray is not None:
                curr_dets = track_with_lk(prev_gray, gray, prev_dets, curr_dets)

            imm, caution, det = evaluate_collision(curr_dets)
            now = time.time()
            if (imm or caution) and (now - last_sent_ts >= 1.0):
                level = "danger" if imm else "caution"
                msg = make_warning(det, level)
                pool.submit(
                    requests.post,
                    SERVER_URL,
                    json={"sentence": msg},
                    timeout=0.5
                )
                print(msg)
                last_sent_ts = now
                if imm:
                    caution_cnt = safe_cnt = 0
                else:
                    caution_cnt = min(caution_cnt + 1, SAFE_FRAME_BUFFER)
            else:
                safe_cnt += 1
                if safe_cnt >= SAFE_FRAME_BUFFER:
                    caution_cnt = safe_cnt = 0

            prev_gray, prev_dets = gray, curr_dets

            spare = FRAME_INTERVAL - (time.time() - loop_start)
            if spare > 0:  time.sleep(spare)

# ==============================
# 5. 메인
# ==============================
def main():
    cap = cv2.VideoCapture(CAM_URL, cv2.CAP_ANY)
    if not cap.isOpened():
        print("Error: Camera open failed.")
        return

    q = Queue(maxsize=1)
    stop_evt = threading.Event()

    t_cap  = threading.Thread(target=capture_worker,   args=(cap, q), daemon=True)
    t_proc = threading.Thread(target=processing_worker, args=(q, stop_evt), daemon=True)
    t_cap.start(); t_proc.start()

    try:
        while t_proc.is_alive():
            t_proc.join(timeout=0.5)
    except KeyboardInterrupt:
        print("Stopping…")
    finally:
        stop_evt.set()
        t_cap.join(); t_proc.join()

if __name__ == "__main__":
    main()
